if [ "$USER" == "pratool" ]
then
    ../usrp_examples/tx_samples_from_file --freq 2.4885e9 --rate 0.25e6 --type float --file ./sent.dat --args="olin_usrp11"
else
    ../examples/tx_samples_from_file --freq 2.4885e9 --rate 0.25e6 --type float --file ./sent.dat --args="olin_usrp11"
fi
