import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from realrobot.env import Robot

import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R


def parse_args():
    parser = argparse.ArgumentParser(description="Publish robot TCP pose as PoseStamped.")
    parser.add_argument("--robot-ip", default="192.168.58.6", help="FR5 robot IP address.")
    parser.add_argument("--topic", default="robot_pose", help="ROS topic to publish.")
    parser.add_argument("--frame-id", default="base_link", help="Pose frame id.")
    parser.add_argument("--rate", type=float, default=30.0, help="Publish rate in Hz.")
    return parser.parse_args()


def main():
    args = parse_args()

    rospy.init_node("robot_pose_publisher", anonymous=True)
    pub = rospy.Publisher(args.topic, PoseStamped, queue_size=10)

    robot = Robot.RPC(args.robot_ip)
    rate = rospy.Rate(args.rate)

    while not rospy.is_shutdown():
        error = robot.GetActualTCPPose()
        if isinstance(error, int):
            rospy.logwarn("Failed to read TCP pose, error code: %s", error)
        else:
            current_pose = error[1]
            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = args.frame_id

            pose_msg.pose.position.x = current_pose[0] / 1000.0
            pose_msg.pose.position.y = current_pose[1] / 1000.0
            pose_msg.pose.position.z = current_pose[2] / 1000.0

            r = R.from_euler("xyz", current_pose[3:6], degrees=True)
            q = r.as_quat()
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]

            pub.publish(pose_msg)

        rate.sleep()


if __name__ == "__main__":
    main()
