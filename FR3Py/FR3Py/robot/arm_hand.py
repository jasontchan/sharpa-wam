from FR3Py.robot.interface import FR3Real


class FR3WithHand:
    """
    Class that composes the FR3 arm interface and the Sharpa Wave hand
    interface behind a single API so that both can be driven from one process.
    """

    def __init__(self, robot_id="franka", enable_hand=True, hand=None, **hand_kwargs):
        """
        Initialize an instance of the class.

        @param robot_id: (str) The unique ID of the arm, used to generate the
        LCM message topic names. Defaults to "franka".
        @param enable_hand: (bool) Whether to construct the hand interface.
        @param hand: (SharpaHand or None) An already constructed hand interface
        to use instead of building a new one.
        @param hand_kwargs: Keyword arguments forwarded to `SharpaHand`.
        """
        self.arm = FR3Real(robot_id=robot_id)
        self.hand = hand
        if self.hand is None and enable_hand:
            self.hand = self.connectHand(**hand_kwargs)

    def connectHand(self, **hand_kwargs):
        """
        Construct the hand interface.

        @param hand_kwargs: Keyword arguments forwarded to `SharpaHand`.
        @return: (SharpaHand) The connected hand interface.
        """
        from FR3Py.robot.hand_interface import SharpaHand

        self.hand = SharpaHand(**hand_kwargs)
        return self.hand

    def getJointStates(self):
        """
        Get the current state of the arm and of the hand.

        @return: (dict) {'arm': the arm joint state or None,
        'hand': the hand joint state or None}.
        """
        return {
            "arm": self.arm.getJointStates(),
            "hand": self.hand.getJointStates() if self.hand is not None else None,
        }

    def getTactile(self):
        """
        Get the latest tactile reading of the hand.

        @return: (dict or None) The per-channel tactile reading, or None when
        no hand is attached.
        """
        return self.hand.getTactile() if self.hand is not None else None

    def setCommands(self, arm_cmd=None, hand_cmd=None):
        """
        Send a command to the arm, to the hand, or to both.

        @param arm_cmd: (numpy.ndarray, shape=(7,) or None) The arm command.
        @param hand_cmd: (numpy.ndarray, shape=(22,) or None) The hand command.
        """
        if arm_cmd is not None:
            self.arm.setCommands(arm_cmd)
        if hand_cmd is not None:
            if self.hand is None:
                raise RuntimeError("No hand interface is attached.")
            self.hand.setCommands(hand_cmd)

    def goHome(self):
        """
        Drive the hand to its home configuration. The arm is left untouched.
        """
        if self.hand is not None:
            self.hand.goHome()

    def close(self):
        """
        Shut down both the arm and the hand interfaces.
        """
        self.arm.close()
        if self.hand is not None:
            self.hand.close()
            self.hand = None
