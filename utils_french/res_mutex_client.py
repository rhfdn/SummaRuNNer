import socket

# read_line from socket
def read_line_from_socket(s):
  msg = b""

  while True:
    f = s.recv(1)
    if not f:
      return None
    if f == b"\n":
      break
    msg += f

  return msg.decode("utf-8")

class ResMutexClient():
  def __init__(self, ip="localhost", port_num=5002) -> None:
    self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.client_socket.connect((ip, port_num))
    self.client_id = read_line_from_socket(self.client_socket)
  
  def lock(self, res_id):
    print("lock", res_id)
    self.client_socket.sendall(("lock " + res_id + "\n").encode())

    if read_line_from_socket(self.client_socket) != "Ok":
      raise Exception("Error on lock", res_id)
  
  def unlock(self, res_id):
    print("unlock", res_id)
    self.client_socket.sendall(("unlock " + res_id + "\n").encode())

    if read_line_from_socket(self.client_socket) != "Ok":
      raise Exception("Error on unlock", res_id)

  def close(self):
    print("close (exit)")
    self.client_socket.sendall("exit\n".encode())

    if read_line_from_socket(self.client_socket) != "Ok":
      raise Exception("Error on exit")
