# Legal MemSum

<img title="a title" alt="Alt text" src="robot-judge.jpg" width="300vw">

## Download the code

You can download the code using git with the following command

    git clone git@github.com:bauerem/legal_memsum.git

Navigate into the created repository with

    cd legal_memsum

## Installation

Setup an environment and download model weights

    bash setup.sh


## Use model

To summarize an opinion, simply run the script "run_memsum.py". The script takes a filename containing an opinion to be summarized as an argument. A sample opinion is provided in the file "opinion.txt". To run the summarization, run the following python command

    python run_memsum.py -f opinion.txt
  
We also have a Huggingface Space, visit it here
    https://huggingface.co/spaces/bauerem/memsum_app

## Output

The script will output the summary as a string of sentences, each starting on a new line, e.g.

    Standard of Review 5 The construction and interpretation of written agreements is a question of law.
    Whether a portion of a written agreement is ambiguous is also a question of law.
    We review questions of law de novo to determine whether they are correct.
    This Court has held that it will ""look to the substance of a motion, not just its 2 In re Marriage of Holloway title, to identify what motion has been presented.""
    Black's Law Dictionary defines ""latent ambiguity"" as ""[a] defect which does not appear on the face of language used or an instrument being considered.
    It arises when language is clear and intelligible and suggests but a single meaning, but some extrinsic fact or some 299 Mont.
    2000) extraneous evidence creates a necessity for interpretation or a choice between two or more possible meanings.""
    360, 366, 742 P.2d 1009, 1013 (citing § 28-3-301, MCA, which states: ""A contract must be so interpreted as to give effect to the mutual intention of the parties as it existed at the time of contracting, so far as the same is ascertainable and lawful."").
    We have previously held that ""`[t]he interpretation or clarification of an ambiguous judgment does not involve amendment thereof, so that even though power to modify is lacking, a court may construe and clarify a decree disposing of property, or enforce it.'""
