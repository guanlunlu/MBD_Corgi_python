function PhiRL = getPhiRL_simulink(sm_PhiRL)
%Convert singleboard rio phi frame to simulink frame
phiR = deg2rad(90-17) - sm_PhiRL(1,1);
phiL = deg2rad(90+17) - sm_PhiRL(2,1);
PhiRL = [phiR; phiL];
end