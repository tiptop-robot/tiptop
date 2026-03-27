from tiptop.utils import get_robot_client


def get_joint_positions():
    client = get_robot_client()
    print(client.get_joint_positions())
