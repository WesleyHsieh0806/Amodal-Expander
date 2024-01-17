LVIS="/data3/chengyeh/lvis"
mkdir -p $LVIS
# download lvis val/train annotations
wget "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip" -O $LVIS/lvis_v1_val.json.zip
wget "https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip" -O $LVIS/lvis_v1_train.json.zip

unzip $LVIS/lvis_v1_val.json.zip -d $LVIS
unzip $LVIS/lvis_v1_train.json.zip -d $LVIS

rm $LVIS/lvis_v1_val.json.zip
rm $LVIS/lvis_v1_train.json.zip