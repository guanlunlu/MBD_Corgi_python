function SimulinkPhiRL = getSimulinkPhiRL(PhiRL)
%Convert singleboard rio phi frame to simulink frame
simu_phiR = deg2rad(90-17) - PhiRL(1,1);
simu_phiL = deg2rad(90+17) - PhiRL(2,1);
SimulinkPhiRL = [simu_phiR; simu_phiL];
end