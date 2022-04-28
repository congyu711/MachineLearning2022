from socket import *
serverPort = 12000
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind(('', serverPort))
serverSocket.listen(1)
print("The server is ready to receive")
connectionSocket, addr=serverSocket.accept()
while True:
    sentence=connectionSocket.recv(4096).decode()
    capsen=sentence.upper()
    connectionSocket.send(capsen.encode())
