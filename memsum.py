"""
@author: Nianlong Gu, Institute of Neuroinformatics, ETH Zurich
@email: nianlonggu@gmail.com

The source code for the paper: MemSum: Extractive Summarization of Long Documents using Multi-step Episodic Markov Decision Processes

When using this code or some of our pre-trained models for your application, please cite the following paper:
@article{DBLP:journals/corr/abs-2107-08929,
  author    = {Nianlong Gu and
               Elliott Ash and
               Richard H. R. Hahnloser},
  title     = {MemSum: Extractive Summarization of Long Documents using Multi-step
               Episodic Markov Decision Processes},
  journal   = {CoRR},
  volume    = {abs/2107.08929},
  year      = {2021},
  url       = {https://arxiv.org/abs/2107.08929},
  eprinttype = {arXiv},
  eprint    = {2107.08929},
  timestamp = {Thu, 22 Jul 2021 11:14:11 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2107-08929.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
import numpy as np

class AddMask( nn.Module ):
    def __init__( self, pad_index ):
        super().__init__()
        self.pad_index = pad_index
    def forward( self, x):
        # here x is a batch of input sequences (not embeddings) with the shape of [ batch_size, seq_len]
        mask = x == self.pad_index
        return mask


class PositionalEncoding( nn.Module ):
    def __init__(self,  embed_dim, max_seq_len = 512  ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        pe = torch.zeros( 1, max_seq_len,  embed_dim )
        for pos in range( max_seq_len ):
            for i in range( 0, embed_dim, 2 ):
                pe[ 0, pos, i ] = math.sin( pos / ( 10000 ** ( i/embed_dim ) )  )
                if i+1 < embed_dim:
                    pe[ 0, pos, i+1 ] = math.cos( pos / ( 10000** ( i/embed_dim ) ) )
        self.register_buffer( "pe", pe )
        ## register_buffer can register some variables that can be saved and loaded by state_dict, but not trainable since not accessible by model.parameters()
    def forward( self, x ):
        return x + self.pe[ :, : x.size(1), :]



class MultiHeadAttention( nn.Module ):
    def __init__(self, embed_dim, num_heads ):
        super().__init__()
        dim_per_head = int( embed_dim/num_heads )
        
        self.ln_q = nn.Linear( embed_dim, num_heads * dim_per_head )
        self.ln_k = nn.Linear( embed_dim, num_heads * dim_per_head )
        self.ln_v = nn.Linear( embed_dim, num_heads * dim_per_head )

        self.ln_out = nn.Linear( num_heads * dim_per_head, embed_dim )

        self.num_heads = num_heads
        self.dim_per_head = dim_per_head
    
    def forward( self, q,k,v, mask = None):
        q = self.ln_q( q )
        k = self.ln_k( k )
        v = self.ln_v( v )

        q = q.view( q.size(0), q.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )
        k = k.view( k.size(0), k.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )
        v = v.view( v.size(0), v.size(1),  self.num_heads, self.dim_per_head  ).transpose( 1,2 )

        a = self.scaled_dot_product_attention( q,k, mask )
        new_v = a.matmul(v)
        new_v = new_v.transpose( 1,2 ).contiguous()
        new_v = new_v.view( new_v.size(0), new_v.size(1), -1 )
        new_v = self.ln_out(new_v)
        return new_v

    def scaled_dot_product_attention( self, q, k, mask = None ):
        ## note the here q and k have converted into multi-head mode 
        ## q's shape is [ Batchsize, num_heads, seq_len_q, dim_per_head ]
        ## k's shape is [ Batchsize, num_heads, seq_len_k, dim_per_head ]
        # scaled dot product
        a = q.matmul( k.transpose( 2,3 ) )/ math.sqrt( q.size(-1) )
        # apply mask (either padding mask or seqeunce mask)
        if mask is not None:
            a = a.masked_fill( mask.unsqueeze(1).unsqueeze(1) , -1e9 )  
        # apply softmax, to get the likelihood as attention matrix
        a = F.softmax( a, dim=-1 )
        return a

class FeedForward( nn.Module ):
    def __init__( self, embed_dim, hidden_dim ):
        super().__init__()
        self.ln1 = nn.Linear( embed_dim, hidden_dim )
        self.ln2 = nn.Linear( hidden_dim, embed_dim )
    def forward(  self, x):
        net = F.relu(self.ln1(x))
        out = self.ln2(net)
        return out


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim ):
        super().__init__()
        self.mha = MultiHeadAttention( embed_dim, num_heads  )
        self.norm1 = nn.LayerNorm( embed_dim )
        self.feed_forward = FeedForward( embed_dim, hidden_dim )
        self.norm2 = nn.LayerNorm( embed_dim )
    def forward( self, x, mask, dropout_rate = 0. ):
        short_cut = x
        net = F.dropout(self.mha( x,x,x, mask ), p = dropout_rate)
        net = self.norm1( short_cut + net )
        short_cut = net
        net = F.dropout(self.feed_forward( net ), p = dropout_rate )
        net = self.norm2( short_cut + net )
        return net

class TransformerDecoderLayer( nn.Module ):
    def __init__(self, embed_dim, num_heads, hidden_dim ):
        super().__init__()
        self.masked_mha = MultiHeadAttention(  embed_dim, num_heads )
        self.norm1 = nn.LayerNorm( embed_dim )
        self.mha = MultiHeadAttention( embed_dim, num_heads )
        self.norm2 = nn.LayerNorm( embed_dim )
        self.feed_forward = FeedForward( embed_dim, hidden_dim )
        self.norm3 = nn.LayerNorm( embed_dim )
    def forward(self, encoder_output, x, src_mask, trg_mask , dropout_rate = 0. ):
        short_cut = x
        net = F.dropout(self.masked_mha( x,x,x, trg_mask ), p = dropout_rate)
        net = self.norm1( short_cut + net )
        short_cut = net
        net = F.dropout(self.mha( net, encoder_output, encoder_output, src_mask ), p = dropout_rate)
        net = self.norm2( short_cut + net )
        short_cut = net
        net = F.dropout(self.feed_forward( net ), p = dropout_rate)
        net = self.norm3( short_cut + net )
        return net 

class MultiHeadPoolingLayer( nn.Module ):
    def __init__( self, embed_dim, num_heads  ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_per_head = int( embed_dim/num_heads )
        self.ln_attention_score = nn.Linear( embed_dim, num_heads )
        self.ln_value = nn.Linear( embed_dim,  num_heads * self.dim_per_head )
        self.ln_out = nn.Linear( num_heads * self.dim_per_head , embed_dim )
    def forward(self, input_embedding , mask=None):
        a = self.ln_attention_score( input_embedding )
        v = self.ln_value( input_embedding )
        
        a = a.view( a.size(0), a.size(1), self.num_heads, 1 ).transpose(1,2)
        v = v.view( v.size(0), v.size(1),  self.num_heads, self.dim_per_head  ).transpose(1,2)
        a = a.transpose(2,3)
        if mask is not None:
            a = a.masked_fill( mask.unsqueeze(1).unsqueeze(1) , -1e9 ) 
        a = F.softmax(a , dim = -1 )

        new_v = a.matmul(v)
        new_v = new_v.transpose( 1,2 ).contiguous()
        new_v = new_v.view( new_v.size(0), new_v.size(1) ,-1 ).squeeze(1)
        new_v = self.ln_out( new_v )
        return new_v


class LocalSentenceEncoder( nn.Module ):
    def __init__( self, vocab_size, pad_index, embed_dim, num_heads , hidden_dim , num_enc_layers , pretrained_word_embedding ):
        super().__init__()
        self.addmask = AddMask( pad_index )
      
        self.rnn = nn.LSTM(  embed_dim, embed_dim, 2, batch_first = True, bidirectional = True)
        self.mh_pool = MultiHeadPoolingLayer( 2*embed_dim, num_heads )
        self.norm_out = nn.LayerNorm( 2*embed_dim )
        self.ln_out = nn.Linear( 2*embed_dim, embed_dim )

        if pretrained_word_embedding is not None:
            ## make sure the pad embedding is 0
            pretrained_word_embedding[pad_index] = 0
            self.register_buffer( "word_embedding", torch.from_numpy( pretrained_word_embedding ) )
        else:
            self.register_buffer( "word_embedding", torch.randn( vocab_size, embed_dim ) )

    """
    input_seq 's shape:  batch_size x seq_len 
    """
    def forward( self, input_seq, dropout_rate = 0. ):
        mask = self.addmask( input_seq )
        ## batch_size x seq_len x embed_dim
        net = self.word_embedding[ input_seq ]
        net, _ = self.rnn( net )
        net =  self.ln_out(F.relu(self.norm_out(self.mh_pool( net, mask ))))
        return net


class GlobalContextEncoder(nn.Module):
    def __init__(self, embed_dim,  num_heads, hidden_dim, num_dec_layers ):
        super().__init__()
        # self.pos_encode = PositionalEncoding( embed_dim)
        # self.layer_list = nn.ModuleList( [  TransformerEncoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range(num_dec_layers) ] )
        self.rnn = nn.LSTM(  embed_dim, embed_dim, 2, batch_first = True, bidirectional = True)
        self.norm_out = nn.LayerNorm( 2*embed_dim )
        self.ln_out = nn.Linear( 2*embed_dim, embed_dim )

    def forward(self, sen_embed, doc_mask, dropout_rate = 0.):
        net, _ = self.rnn( sen_embed )
        net = self.ln_out(F.relu( self.norm_out(net) ) )
        return net


class ExtractionContextDecoder( nn.Module ):
    def __init__( self, embed_dim,  num_heads, hidden_dim, num_dec_layers ):
        super().__init__()
        self.layer_list = nn.ModuleList( [  TransformerDecoderLayer( embed_dim, num_heads, hidden_dim ) for _ in range(num_dec_layers) ] )
    ## remaining_mask: set all unextracted sen indices as True
    ## extraction_mask: set all extracted sen indices as True
    def forward( self, sen_embed, remaining_mask, extraction_mask, dropout_rate = 0. ):
        net = sen_embed
        for layer in self.layer_list:
            #  encoder_output, x,  src_mask, trg_mask , dropout_rate = 0.
            net = layer( sen_embed, net, remaining_mask, extraction_mask, dropout_rate )
        return net

class Extractor( nn.Module ):
    def __init__( self, embed_dim, num_heads ):
        super().__init__()
        self.norm_input = nn.LayerNorm( 3*embed_dim  )
        
        self.ln_hidden1 = nn.Linear(  3*embed_dim, 2*embed_dim  )
        self.norm_hidden1 = nn.LayerNorm( 2*embed_dim  )
        
        self.ln_hidden2 = nn.Linear(  2*embed_dim, embed_dim  )
        self.norm_hidden2 = nn.LayerNorm( embed_dim  )

        self.ln_out = nn.Linear(  embed_dim, 1 )

        self.mh_pool = MultiHeadPoolingLayer( embed_dim, num_heads )
        self.norm_pool = nn.LayerNorm( embed_dim  )
        self.ln_stop = nn.Linear(  embed_dim, 1 )

        self.mh_pool_2 = MultiHeadPoolingLayer( embed_dim, num_heads )
        self.norm_pool_2 = nn.LayerNorm( embed_dim  )
        self.ln_baseline = nn.Linear(  embed_dim, 1 )

    def forward( self, sen_embed, relevance_embed, redundancy_embed , extraction_mask, dropout_rate = 0. ):
        if redundancy_embed is None:
            redundancy_embed = torch.zeros_like( sen_embed )
        net = self.norm_input( F.dropout( torch.cat( [ sen_embed, relevance_embed, redundancy_embed ], dim = 2 ) , p = dropout_rate  )  ) 
        net = F.relu( self.norm_hidden1( F.dropout( self.ln_hidden1( net ) , p = dropout_rate  )   ))
        hidden_net = F.relu( self.norm_hidden2( F.dropout( self.ln_hidden2( net)  , p = dropout_rate  )  ))
        
        p = self.ln_out( hidden_net ).sigmoid().squeeze(2)

        net = F.relu( self.norm_pool(  F.dropout( self.mh_pool( hidden_net, extraction_mask) , p = dropout_rate  )  ))
        p_stop = self.ln_stop( net ).sigmoid().squeeze(1)

        net = F.relu( self.norm_pool_2(  F.dropout( self.mh_pool_2( hidden_net, extraction_mask ) , p = dropout_rate  )  ))
        baseline = self.ln_baseline(net)

        return p, p_stop, baseline


## naive tokenizer with just lower() function
class SentenceTokenizer:
    def __init__(self ):
        pass
    def tokenize(self, sen ):
        return sen.lower()


class Vocab:
    def __init__(self, words, eos_token = "<eos>", pad_token = "<pad>", unk_token = "<unk>" ):
        self.words = words
        self.index_to_word = {}
        self.word_to_index = {}
        for idx in range( len(words) ):
            self.index_to_word[ idx ] = words[idx]
            self.word_to_index[ words[idx] ] = idx
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.eos_index = self.word_to_index[self.eos_token]
        self.pad_index = self.word_to_index[self.pad_token]

        self.tokenizer = SentenceTokenizer()   

    def index2word( self, idx ):
        return self.index_to_word.get( idx, self.unk_token)
    def word2index( self, word ):
        return self.word_to_index.get( word, -1 )
    # The sentence needs to be tokenized 
    def sent2seq( self, sent, max_len = None , tokenize = True):
        if tokenize:
            sent = self.tokenizer.tokenize(sent)
        seq = []
        for w in sent.split():
            if w in self.word_to_index:
                seq.append( self.word2index(w) )
        if max_len is not None:
            if len(seq) >= max_len:
                seq = seq[:max_len -1]
                seq.append( self.eos_index )
            else:
                seq.append( self.eos_index )
                seq += [ self.pad_index ] * ( max_len - len(seq) )
        return seq
    def seq2sent( self, seq ):
        sent = []
        for i in seq:
            if i == self.eos_index or i == self.pad_index:
                break
            sent.append( self.index2word(i) )
        return " ".join(sent)



class MemSum:
    def __init__( self, model_path, vocabulary_path, gpu = None ):
        
        ## max_doc_len is used to truncate too long sentence into at most 100 words 
        max_seq_len =100
        ## max_doc_len is used to truncate too long document into at most 200 sentences 
        max_doc_len = 200
        
        ## These parameters below have been fintuned for the pretrained model
        embed_dim=200
        num_heads=8
        hidden_dim = 1024
        N_enc_l = 2
        N_enc_g = 2
        N_dec = 3
        
        with open( vocabulary_path , "rb" ) as f:
            words = pickle.load(f)
        self.vocab = Vocab( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.global_context_encoder = GlobalContextEncoder( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor( embed_dim, num_heads )
        ckpt = torch.load( model_path, map_location = "cpu" )
        self.local_sentence_encoder.load_state_dict( ckpt["local_sentence_encoder"] )
        self.global_context_encoder.load_state_dict( ckpt["global_context_encoder"] )
        self.extraction_context_decoder.load_state_dict( ckpt["extraction_context_decoder"] )
        self.extractor.load_state_dict(ckpt["extractor"])
        
        self.device =  torch.device( "cuda:%d"%(gpu) if gpu is not None and torch.cuda.is_available() else "cpu"  )        
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
    
    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( "_".join( w_list[ pos:pos+n] )  )
        return ngram_set

    def extract( self, document_batch, p_stop_thres , ngram_blocking , ngram, return_sentence_position, return_sentence_score_history, max_extracted_sentences_per_document ):
        """document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        """
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [""] * ( max_document_length -  len(document) )

            doc_mask.append(  [ 1 if sen.strip() == "" else 0 for sen in  document   ] )

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        extracted_sentences = []
        sentence_score_history = []
        p_stop_history = []
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )
            relevance_embed = self.global_context_encoder( sen_embed, doc_mask  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(np.bool_) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(np.bool_) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]
                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]
                current_redundancy_embed = None
        
                current_hyps = []
                extracted_sen_ngrams = set()

                sentence_score_history_for_doc_i = []

                p_stop_history_for_doc_i = []
                
                for step in range( max_extracted_sentences_per_document+1 ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder( current_sen_embed, current_remaining_mask, current_extraction_mask  )
                    p, p_stop, _ = self.extractor( current_sen_embed, current_relevance_embed, current_redundancy_embed , current_extraction_mask  )
                    p_stop = p_stop.unsqueeze(1)
            
            
                    p = p.masked_fill( current_extraction_mask, 1e-12 ) 

                    sentence_score_history_for_doc_i.append( p.detach().cpu().numpy() )

                    p_stop_history_for_doc_i.append(  p_stop.squeeze(1).item() )

                    normalized_p = p / p.sum(dim=1, keepdims = True)

                    stop = p_stop.squeeze(1).item()> p_stop_thres #and step > 0
                    
                    #sen_i = normalized_p.argmax(dim=1)[0]
                    _, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
                    sorted_sen_indices = sorted_sen_indices[0]
                    
                    extracted = False
                    for sen_i in sorted_sen_indices:
                        sen_i = sen_i.item()
                        if sen_i< len(document_batch[doc_i]):
                            sen = document_batch[doc_i][sen_i]
                        else:
                            break
                        sen_ngrams = self.get_ngram( sen.lower().split(), ngram )
                        if not ngram_blocking or len( extracted_sen_ngrams &  sen_ngrams ) < 1:
                            extracted_sen_ngrams.update( sen_ngrams )
                            extracted = True
                            break
                                        
                    if stop or step == max_extracted_sentences_per_document or not extracted:
                        extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])    ] )
                        extracted_sentences_positions.append( [ sen_i for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])  ]  )
                        break
                    else:
                        current_hyps.append(sen_i)
                        current_extraction_mask_np[0, sen_i] = True
                        current_remaining_mask_np[0, sen_i] = False

                sentence_score_history.append(sentence_score_history_for_doc_i)
                p_stop_history.append( p_stop_history_for_doc_i )
        

        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if return_sentence_score_history:
            results+=[sentence_score_history , p_stop_history ]
        if len(results) == 1:
            results = results[0]
        
        return results
    
    ## document is a list of sentences
    def summarize( self, document, p_stop_thres = 0.7, max_extracted_sentences_per_document = 10, return_sentence_position = False ):
        sentences, sentence_positions = self.extract( [document], p_stop_thres, ngram_blocking = False, ngram = 0, return_sentence_position = True, return_sentence_score_history = False, max_extracted_sentences_per_document = max_extracted_sentences_per_document )
        try:
            sentences, sentence_positions = list(zip(*sorted( zip( sentences[0], sentence_positions[0] ), key = lambda x:x[1] )))
        except ValueError:
            sentences, sentence_positions = (), ()
        if return_sentence_position:
            return sentences, sentence_positions
        else:
            return sentences
       
        
