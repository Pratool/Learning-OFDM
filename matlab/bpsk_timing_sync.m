function [yI, yQ, freq_offset]  = bpsk_timing_sync(xI, xQ)
%function [yI, yQ, freq_offset]  = bpsk_timing_sync(xI, xQ)
% Correct for carrier frequency offset between Tx/Rx for BPSK signals
% using squaring method for frequency estimation. 
%
% Inputs:  xI - I channel samples
%          xQ - Q channel samples
% Outputs: yI - frequency corrected I channel samples
%          yQ - frequency corrected Q channel samples
%          freq_offset - estimated frequency offset
%
% Note that this method of carrier synchronization is accurate up to a
% phase offset of an integer multiple of pi/2. In other words, the 
% transmitted I channel could show up on the Q channel of the receiver 
% and/or the received signal could be  a negated version of the transmitted 
% signal.
% Typically a known header is transmitted as part of a packet before the
% payload bits are transmitted and the known bits can be used to determine
% the true transmitted bits.
% 
%