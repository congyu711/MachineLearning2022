from socket import *
serverName = 'localhost'
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_DGRAM)
message = r'udp\nasldkfjlkajd\nalsdjf\talsdkjf'
clientSocket.sendto(message.encode(), (serverName, serverPort))
modifiedMessage, serverAddress = clientSocket.recvfrom(1024)
print(modifiedMessage.decode())
print(serverAddress)
clientSocket.close()
