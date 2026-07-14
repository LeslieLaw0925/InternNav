import threading
import requests
import time


class EdgeMonitor:
    def __init__(
        self,
        edge_url,
        interval=0.5,
        timeout=0.3
    ):
        self.edge_url = edge_url
        self.interval = interval
        self.timeout = timeout

        # shared states
        self.connected = True
        self.latency = None
        self.last_success_time = None
        self.running = False

    def start(self):
        self.running = True

        thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )
        thread.start()

    def stop(self):
        self.running = False

    def _monitor(self):
        while self.running:
            start = time.time()
            try:
                response = requests.get(
                    self.edge_url,
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    latency = time.time() - start

                    self.connected = True
                    self.latency = latency
                    self.last_success_time = time.time()
                else:
                    self.connected = False

            except Exception:
                self.connected = False
                self.latency = None

            time.sleep(self.interval)