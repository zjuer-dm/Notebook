# ros


## 节点与节点管理器
**节点（Node）— 执行单元**

1. 节点是具体的执行单元，执行具体任务的进程，独立运行的可执行文件

2. 不同的节点可以使用不同的编程语言，可分布式运行在不同的主机

3. 节点在系统中的名称必须是唯一的，重名ros系统会找不到

**节点管理器（ROS Master) — 控制中心**

1. 节点的管理。命名、注册、建立通讯

2. 提供参数服务器，节点使用此服务器存储和检索运行时的参数

**话题通信（异步）**

话题 topic

1. 分为发布者和订阅者

2. 单向数据传输，从驱动端传到订阅段，发布者到订阅者

3. 通道被定义为话题，时数据传输的总线

**消息 message**

1. 具有一定的类型和数据结构（有ros提供的标准类型和用户自定义的类型

2. 使用与编程语言无关的 .msg 文件定义类型和数据结构

**服务通信（同步） 服务 service**

1. 使用客户端/服务器（service/client) 模型，客户端发送请求数据，服务器完成处理后返回应答数据

2. 使用与编程语言无关的, .srv 文件定义请求和应答的数据结构

3. 一般是一次，发出一个配置指令，完成配置内容，返回一个反馈

## ROS命令行工具
roscore启动ROS master(要首先运行的一个指令）

rqt_graph用来显示系统计算图的工具

rosnode list：列出系统当中使用的节点 ， rosout是ros默认的一个话题

rosnode info **： 查看某个节点的信息 ，发布和订阅的信息  rosnode info /turtlesim

rosmsg show显示消息的数据结构

rosservice list ：提供的所有服务 ，服务端都是仿真器 ，客户端是终端

rosbag record -a -O cmd_record：话题记录

rosbag play cmd_record.bag：话题复现

## 创建工作空间与功能包
工作空间（workspace)是一个存放工程开发相关文件的文件夹。所有的源码、配置文件、可执行文件都是放置在里面的。主要分为四个文件夹：
1. src: 代码空间 (source space)

放置功能包（里面的代码、配置文件、launch文件
2. build:编译空间 (build space)

编译过程中的中间文件（不太用关心
3. devel:开发空间 (development space)

编译生成的一些可执行文件、库、脚本
4. install:安装空间 (install space）

install命令的结果就放在里面

![alt text](<pic/Screenshot 2024-11-14 at 15.57.14.png>)

![alt text](<pic/Screenshot 2024-11-14 at 16.35.41.png>)
创建功能包：

```bash 
catkin_create_pkg test_pkg std_msgs rospy roscpp
```
## Publisher的编程实现

![alt text](<pic/Screenshot 2024-11-14 at 16.30.44.png>)

**必须注意，我们设置的node名称，发布的topic**

构建功能包：

在src目录下：

```bash
catkin_create_pkg learning_topic rospy roscpp std_msgs geometry_msgs turtlesim
```

```c++
#include <ros/ros.h>
#include <geometry_msgs/Twist.h>

int main(int argc, char **argv)
{
	// ROS节点初始化
	ros::init(argc, argv, "velocity_publisher");

	// 创建节点句柄
	ros::NodeHandle n;

	// 创建一个Publisher，发布名为/turtle1/cmd_vel的topic，消息类型为geometry_msgs::Twist，队列长度10
	ros::Publisher turtle_vel_pub = n.advertise<geometry_msgs::Twist>("/turtle1/cmd_vel", 10);

	// 设置循环的频率
	ros::Rate loop_rate(10);

	int count = 0;
	while (ros::ok())
	{
	    // 初始化geometry_msgs::Twist类型的消息
		geometry_msgs::Twist vel_msg;
		vel_msg.linear.x = 0.5;
		vel_msg.angular.z = 0.2;

	    // 发布消息
		turtle_vel_pub.publish(vel_msg);
		ROS_INFO("Publsh turtle velocity command[%0.2f m/s, %0.2f rad/s]", 
				vel_msg.linear.x, vel_msg.angular.z);

	    // 按照循环频率延时
	    loop_rate.sleep();
	}

	return 0;
}
```
其中python出现bug:

1. chmod +x  确保你的 Python 脚本是可执行的

2. 添加 Shebang（解释器声明）
首先，在 veo_publisher.py 文件的最顶部添加一行 shebang，告诉系统这是一个 Python 脚本。


```py
import rospy
from geometry_msgs.msg import Twist

def velocity_publisher():
	# ROS节点初始化
    rospy.init_node('velocity_publisher', anonymous=True)

	# 创建一个Publisher，发布名为/turtle1/cmd_vel的topic，消息类型为geometry_msgs::Twist，队列长度10
    turtle_vel_pub = rospy.Publisher('/turtle1/cmd_vel', Twist, queue_size=10)

	#设置循环的频率
    rate = rospy.Rate(10) 

    while not rospy.is_shutdown():
		# 初始化geometry_msgs::Twist类型的消息
        vel_msg = Twist()
        vel_msg.linear.x = 0.5
        vel_msg.angular.z = 0.2

		# 发布消息
        turtle_vel_pub.publish(vel_msg)
    	rospy.loginfo("Publsh turtle velocity command[%0.2f m/s, %0.2f rad/s]", 
				vel_msg.linear.x, vel_msg.angular.z)

		# 按照循环频率延时
        rate.sleep()

if __name__ == '__main__':
    try:
        velocity_publisher()
    except rospy.ROSInterruptException:
        pass
    
```
代码编译规则：
![alt text](<pic/Screenshot 2024-11-14 at 16.50.48.png>)

## subscriber：

```c++
#include <ros/ros.h>
#include "turtlesim/Pose.h"

// 接收到订阅的消息后，会进入消息回调函数
void poseCallback(const turtlesim::Pose::ConstPtr& msg)
{
    // 将接收到的消息打印出来
    ROS_INFO("Turtle pose: x:%0.6f, y:%0.6f", msg->x, msg->y);
}

int main(int argc, char **argv)
{
    // 初始化ROS节点
    ros::init(argc, argv, "pose_subscriber");

    // 创建节点句柄
    ros::NodeHandle n;

    // 创建一个Subscriber，订阅名为/turtle1/pose的topic，注册回调函数poseCallback
    ros::Subscriber pose_sub = n.subscribe("/turtle1/pose", 10, poseCallback);

    // 循环等待回调函数
    ros::spin();

    return 0;
}
```

```py
import rospy
from turtlesim.msg import Pose

def poseCallback(msg):
    rospy.loginfo("Turtle pose: x:%0.6f, y:%0.6f", msg.x, msg.y)

def pose_subscriber():
	# ROS节点初始化
    rospy.init_node('pose_subscriber', anonymous=True)

	# 创建一个Subscriber，订阅名为/turtle1/pose的topic，注册回调函数poseCallback
    rospy.Subscriber("/turtle1/pose", Pose, poseCallback)

	# 循环等待回调函数
    rospy.spin()

if __name__ == '__main__':
    pose_subscriber()

```

![alt text](<pic/Screenshot 2024-11-14 at 20.17.03.png>)

```c++
/**
 * 该例程设置/读取海龟例程中的参数
 */
#include <string>
#include <ros/ros.h>
#include <std_srvs/Empty.h>

int main(int argc, char **argv)
{
	int red, green, blue;

    // ROS节点初始化
    ros::init(argc, argv, "parameter_config");

    // 创建节点句柄
    ros::NodeHandle node;

    // 读取背景颜色参数
	ros::param::get("/background_r", red);
	ros::param::get("/background_g", green);
	ros::param::get("/background_b", blue);

	ROS_INFO("Get Backgroud Color[%d, %d, %d]", red, green, blue);

	// 设置背景颜色参数
	ros::param::set("/background_r", 255);
	ros::param::set("/background_g", 255);
	ros::param::set("/background_b", 255);

	ROS_INFO("Set Backgroud Color[255, 255, 255]");

    // 读取背景颜色参数
	ros::param::get("/background_r", red);
	ros::param::get("/background_g", green);
	ros::param::get("/background_b", blue);

	ROS_INFO("Re-get Backgroud Color[%d, %d, %d]", red, green, blue);

	// 调用服务，刷新背景颜色
	ros::service::waitForService("/clear");
	ros::ServiceClient clear_background = node.serviceClient<std_srvs::Empty>("/clear");
	std_srvs::Empty srv;
	clear_background.call(srv);
	
	sleep(1);

    return 0;
}

```

```py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import rospy
from std_srvs.srv import Empty

def parameter_config():
	# ROS节点初始化
    rospy.init_node('parameter_config', anonymous=True)

	# 读取背景颜色参数
    red   = rospy.get_param('/background_r')
    green = rospy.get_param('/background_g')
    blue  = rospy.get_param('/background_b')

    rospy.loginfo("Get Backgroud Color[%d, %d, %d]", red, green, blue)

	# 设置背景颜色参数
    rospy.set_param("/background_r", 255);
    rospy.set_param("/background_g", 255);
    rospy.set_param("/background_b", 255);

    rospy.loginfo("Set Backgroud Color[255, 255, 255]");

	# 读取背景颜色参数
    red   = rospy.get_param('/background_r')
    green = rospy.get_param('/background_g')
    blue  = rospy.get_param('/background_b')

    rospy.loginfo("Get Backgroud Color[%d, %d, %d]", red, green, blue)

	# 发现/spawn服务后，创建一个服务客户端，连接名为/spawn的service
    rospy.wait_for_service('/clear')
    try:
        clear_background = rospy.ServiceProxy('/clear', Empty)

		# 请求服务调用，输入请求数据
        response = clear_background()
        return response
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e

if __name__ == "__main__":
    parameter_config()
```
## launch文件

通过launch文件以及roslaunch命令可以一次性启动多个节点，并且可以设置丰富的参数。

node标签会指定一个准备运行的ROS节点，node标签是 launch 文件中最重要的标签，因为它实现了launch文件的基本功能，即同时启动多个ROS节点。

```xml
<node pkg="package_name" type="executable_node" name="node_name" args="$()" respawn="true" output="sceen">

```
1. pkg：节点所在功能包的名称package_name；
2. type：节点类型是可执行文件(节点)的名称executable_node；
3. name：节点运行时的名称node_name；
4. args：传递命令行设置的参数；
5. respawn：是否自动重启，true表示如果节点未启动或异常关闭，则自动重启；false则表示不自动重启，默认值为false；
6. output：是否将节点信息输出到屏幕，如果不设置该属性，则节点信息会被写入到日志文件，并不会显示到屏幕上。


param:

param标签则可以实现传递参数的功能，它可以定义一个将要被设置到参数
服务器的参数，它的参数值可以通过文本文件、二进制文件或命令等属性来设置。

```xml
<param name="param_name" type="param_type" value="param_value" />
<!-- param 标签可以嵌入到 node 标签中，以此来作为该 node 的私有参数 -->
<node>
	<param name="param_name" type="param_type" value="param_value" />
</node>
```
1. name：参数名称param_name
2. type：参数类型double，str，int，bool，yaml
3. value：需要设置的参数值param_value

rosparam标签可以实现节点从参数服务器上加载(load)、导出(dump)和删除(delete)YAML文件

```xml
<!-- 加载package_name功能包下的example.yaml文件 -->
<rosparam command="load" file="$(find package_name)/example.yaml">
<!-- 导出example_out.yaml文件到package_name功能包下 -->
<rosparam command="dump" file="$(find package_name)/example_out.yaml" />
<!-- 删除参数 -->
<rosparam command="delete" param="xxx/param">
```

include标签功能和编程语言中的include预处理类似，它可以导入其他launch文件到当前include标签所在的位置，实现launch文件复用。

```xml
<include file="$(find package_name)/launch_file_name">

```
remap标签可以实现节点名称的重映射，每个remap标签包含一个原始名称和一个新名称，在系统运行后原始名称会被替换为新名称。

```xml
<arg name="arg_name" default="arg_default" />
<arg name="arg_name" value="arg_value" />
<!-- 命令行传递的 arg 参数可以覆盖 default，但不能覆盖 value。 -->
```


```xml
<arg name="arg_name" default="arg_default" />
<arg name="arg_name" value="arg_value" />
<!-- 命令行传递的 arg 参数可以覆盖 default，但不能覆盖 value。 -->

```
* arg	启动时的参数，只在launch文件中有意义
* param	运行时的参数，参数会存储在参数服务器中

group标签可以实现将一组配置应用到组内的所有节点，它也具有命名空间ns特点，可以将不同的节点放入不同的 namespace。
```xml

<!-- 用法1 -->
<group ns="namespace_1">
    <node pkg="pkg_name1" .../>
    <node pkg="pkg_name2" .../>
    ...
</group>

<group ns="namespace_2">
    <node pkg="pkg_name3" .../>
    <node pkg="pkg_name4" .../>
    ...
</group>
<!-- 用法2 -->
<!-- if = value：value 为 true 则包含内部信息 -->
<group if="$(arg foo1)">
    <node pkg="pkg_name1" .../>
</group>

<!-- unless = value：value 为 false 则包含内部信息 -->
<group unless="$(arg foo2)">
    <node pkg="pkg_name2" .../>
</group>
<!--
	当 foo1 == true 时包含其标签内部
	当 foo2 == false 时包含其标签内部
-->
```

# ros2
