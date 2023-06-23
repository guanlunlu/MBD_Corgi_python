function PhiRL = getPhiRL(ThetaBeta)
PhiRL = [1 1; -1 1]* ThetaBeta - [1; -1]*deg2rad(17);
end