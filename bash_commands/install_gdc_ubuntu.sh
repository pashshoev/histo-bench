sudo apt-get install unzip -y

wget https://gdc.cancer.gov/system/files/public/file/gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip
unzip gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip
unzip gdc-client_2.3_Ubuntu_x64.zip
sudo mv gdc-client /usr/local/bin
rm gdc-client_2.3_Ubuntu_x64-py3.8-ubuntu-20.04.zip
rm gdc-client_2.3_Ubuntu_x64.zip
rm gdc-client