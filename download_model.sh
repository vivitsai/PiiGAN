DIR='./data_models'
URL='#'

echo "Downloading SEGAN pre-trained models..."
mkdir -p $DIR
FILE="$(curl -sc /tmp/gcokie "${URL}" | grep -o '="uc-name.*</span>' | sed 's/.*">//;s/<.a> .*//')" 
curl -Lb /tmp/gcokie "${URL}&confirm=$(awk '/_warning_/ {print $NF}' /tmp/gcokie)" -o "$DIR/${FILE}" 

echo "Extracting SEGAN pre-trained models..."
cd $DIR
unzip $FILE
rm $FILE

echo "Download success."