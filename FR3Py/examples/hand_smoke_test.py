import sys
import time

import numpy as np

from FR3Py.robot.hand_interface import HandConfigError, SharpaHand

DURATION = 5.0
RATE = 30.0


def report_hand_error(error, hand=None):
    """
    Report an error raised by the hand, along with the fault codes it reports
    when a handle to it is available. The fault codes say more about why a
    command was rejected than a traceback does.

    @param error: (RuntimeError) The error that was raised. `HandConfigError`
    is reported as a configuration failure, anything else as a rejected call.
    @param hand: (SharpaHand or None) The interface, when it was constructed.
    """
    kind = "configuration failed" if isinstance(error, HandConfigError) else "error"
    print(f"[hand_smoke_test] hand {kind}: {error}", file=sys.stderr)
    if hand is None or hand.hand is None:
        print(
            "[hand_smoke_test] no hand handle available, cannot read fault codes.",
            file=sys.stderr,
        )
        return
    print(
        f"[hand_smoke_test] fault codes: {hand.hand.get_fault_code()}",
        file=sys.stderr,
    )


def main():
    """
    Stream joint states and tactile readings for a few seconds and return the
    process exit code.

    @return: (int) 0 on success, 1 when the hand configuration could not be
    confirmed or the hand rejected a call.
    """
    try:
        hand = SharpaHand(hand_side=1, enable_tactile=True)
    except RuntimeError as e:
        report_hand_error(e)
        return 1

    dt = 1.0 / RATE
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < DURATION:
            loop_start = time.perf_counter()

            state = hand.getJointStates()
            tactile = hand.getTactile()
            forces = np.array(
                [tactile[ch]["force"] for ch in sorted(tactile.keys())],
                dtype=np.float32,
            )

            print(f"t={time.perf_counter() - start:5.2f}s")
            print(f"  q       : {np.array2string(state['q'], precision=3)}")
            print(f"  forces  : {np.array2string(forces, precision=3)}")

            remaining = dt - (time.perf_counter() - loop_start)
            if remaining > 0:
                time.sleep(remaining)

        hand.goHome()
    except RuntimeError as e:
        report_hand_error(e, hand)
        return 1
    finally:
        hand.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
