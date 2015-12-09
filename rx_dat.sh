rm ./received.dat
if [ "$USER" == "pratool" ]
then
    ../usrp_examples/rx_samples_to_file --freq 2.4885e9 --rate 0.25e6 --type float --file ./received.dat --args="name=olin_usrp09"
else
    ../examples/rx_samples_to_file --freq 2.4885e9 --rate 0.25e6 --type float --file ./received.dat --args="name=olin_usrp09"
fi
