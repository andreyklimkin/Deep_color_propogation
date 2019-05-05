import subprocess
import threading

""" Run system commands with timeout
"""
class Command(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None
        self.out = None
        self.thread = None

    def run_command(self, capture = False):
        if not capture:
            self.process = subprocess.Popen(self.cmd,shell=True)
            self.process.communicate()
            return
        # capturing the outputs of shell commands
        self.process = subprocess.Popen(self.cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.PIPE)
        out,err = self.process.communicate()
        if len(out) > 0:
            self.out = out.splitlines()
        else:
            self.out = None

    # set default timeout to 2 minutes
    def run(self, capture = False):
        self.thread = threading.Thread(target=self.run_command, args=(capture,))
        self.thread.start()
        #thread.join(timeout)
        #if thread.is_alive():
        #    print('Command timeout, kill it: ' + self.cmd)
        #    self.process.kill()
        #    thread.join()
        #return self.out
    def wait(self, timeout=10):
        self.thread.join(timeout)
        if self.thread.is_alive():
            print('Command timeout, kill it: ' + self.cmd)
            self.process.kill()
            self.thread.join()
