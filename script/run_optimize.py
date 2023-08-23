from cpg_fsm import *

loop_freq = 1000  # Hz

# evo1
bp = [
    0.05,
    0.01477402,
    0.01,
    -0.08332673,
    0.09,
    -0.11168419,
    0.05,
    0.0,
    0.01,
    0.0274304,
    0.01,
    0.05320294,
    0.085,
    0.02,
    0.01,
    0.03632489,
    0.01,
    0.004311,
    0.05,
    0.0,
    0.01,
    0.01088432,
    0.01,
    -0.1,
]

bp = np.array(
    [
        0.05619701,
        0.00411106,
        0.01690634,
        -0.03432745,
        0.08628042,
        -0.09429632,
        0.05316414,
        0.00363167,
        0.02571238,
        0.04847964,
        0.01144883,
        0.08482995,
        0.07099504,
        0.0061735,
        0.04818854,
        0.00514284,
        0.01187029,
        -0.0274283,
        0.05789841,
        0.01448661,
        0.01453597,
        0.0163456,
        0.01975994,
        -0.07417478,
    ]
)

bp = np.array(bp).reshape(4, -1)

FSM = FiniteStateMachine(loop_freq)
CORGI = Corgi(loop_freq, FSM)
CORGI.standUp(0.8)
CORGI.move(swing_profile=bp)
print("after optimize cost", CORGI.cost)
CORGI.visualize()

bez_prof_init = np.array(
    [
        [0.08, 0.01, 0.15, 0.05, 0.08, 0.06],
        [0.08, 0.01, 0.15, 0.05, 0.08, 0.06],
        [0.08, 0.01, 0.08, 0.05, 0.02, 0.1],
        [0.08, 0.01, 0.08, 0.05, 0.02, 0.1],
    ]
)

FSM = FiniteStateMachine(loop_freq)
CORGI = Corgi(loop_freq, FSM)
CORGI.standUp(0.8)
CORGI.move(swing_profile=bez_prof_init)
print("before optimize cost", CORGI.cost)
CORGI.visualize()
