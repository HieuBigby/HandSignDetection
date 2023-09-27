import socket

# Specify the target IP address and port
target_ip = '192.168.1.114'  # Replace with the actual target IP address
target_port = 5052         # Replace with the actual target port

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Data to send
data = "Hello, world!"

# Convert data to bytes
data_bytes = data.encode()

# Use sendto() to send data to the target
while True:
    print('Sending...')
    sock.sendto(data_bytes, (target_ip, target_port))

# # Close the socket
# sock.close()