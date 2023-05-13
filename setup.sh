# Environment setup
python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

# Download weights
wget https://www.dropbox.com/s/t490d918orlit1v/memsum_model.zip
unzip memsum_model.zip
mkdir -p model/MemSum_Final
mv model.pt model/MemSum_Final/model.pt
rm -rf memsum_model.zip