from socket import *
serverName = '20.20.20.128'
serverPort = 12000
clientSocket = socket(AF_INET, SOCK_STREAM)
message = r'sdfsdfdkjf'
clientSocket.connect((serverName,serverPort))
clientSocket.sendto(message.encode(),(serverName, serverPort))
modifiedMessage, serverAddress = clientSocket.recvfrom(2048)
print(modifiedMessage.decode())
print(serverAddress)
clientSocket.close()
