
import queue
import threading
import time

class ConsolidationManager:
    """
    Manages the asynchronous, "background" consolidation of memories for the HTM.
    This realizes the "No-Sleep" model, where consolidation is a continuous
    process rather than a discrete phase.
    """
    def __init__(self, temporal_memory):
        """
        Args:
            temporal_memory (TemporalMemory): The TM module whose memories
                                              will be consolidated.
        """
        self.tm = temporal_memory
        self.consolidation_queue = queue.Queue()
        self.stop_event = threading.Event()

        # The "subconscious" background thread
        self.worker_thread = threading.Thread(
            target=self._consolidation_worker,
            daemon=True
        )
        self.worker_thread.start()
        print("ConsolidationManager: Background worker started.")

    def add_to_consolidation_queue(self, memory_trace):
        """
        Adds a significant memory trace to the queue for consolidation.

        Args:
            memory_trace (list[torch.Tensor]): The sequence of SDRs to be made permanent.
        """
        print("Adding new memory trace to consolidation queue.")
        self.consolidation_queue.put(memory_trace)

    def _consolidation_worker(self):
        """
        The main loop for the background thread. Continuously checks the queue
        and tells the TM to consolidate memories.
        """
        while not self.stop_event.is_set():
            try:
                # Wait for a memory to be added to the queue (with a timeout)
                trace = self.consolidation_queue.get(timeout=1)
                self.tm.consolidate(trace)
                self.consolidation_queue.task_done()
            except queue.Empty:
                # Queue is empty, just continue waiting.
                continue

    def stop(self):
        """Stops the background worker thread gracefully."""
        print("ConsolidationManager: Stopping worker.")
        self.stop_event.set()
        self.worker_thread.join()