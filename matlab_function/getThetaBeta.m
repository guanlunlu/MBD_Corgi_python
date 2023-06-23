function ThetaBeta = getThetaBeta(PhiRL)
ThetaBeta = 1/2 * [1 -1 ; 1 1] * PhiRL + [1; 0] * deg2rad(17);
end