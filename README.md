###"跳转链接"
[跳转链接](https://github.com/InternLM/Tutorial)


#linux
#完成ssh连接与端口映射

#ssh连接，输入命令后输入密码
ssh -p 35332 root@ssh.intern-ai.org.cn -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
![image](https://github.com/user-attachments/assets/ad297a33-5d12-47c7-8cb4-36e310c07dcc)

#配置端口映射
ssh -p 35332 root@ssh.intern-ai.org.cn -CNg -L 7860:127.0.0.1:7860 -o StrictHostKeyChecking=no
成功后如图所示
![image](https://github.com/user-attachments/assets/0ef1a61c-e35b-470b-9b2c-1813d1600137)

#创建hello_world.py

![image](https://github.com/user-attachments/assets/36e942b7-a275-4e4d-aac4-6b57a5a0142f)

#运行hello_world.py
![image](https://github.com/user-attachments/assets/a98ee260-0712-4198-8c48-8d85f61c47ac)
#########################################

#使用 VSCODE 远程连接开发机
![image](https://github.com/user-attachments/assets/b958809a-8246-42f4-a104-840bfae5cfa6)

#创建一个名为test的conda环境
conda create -n test python==3.8
![image](https://github.com/user-attachments/assets/a2936a95-cc10-4c68-a4d3-e1e659e582fd)



#git

#提交破冰介绍

![image](https://github.com/user-attachments/assets/5605ff6d-bdd9-42d2-88ee-2819bbc93f20)
![image](https://github.com/user-attachments/assets/2b87236a-badd-4656-98bc-bbaa68debf9f)




