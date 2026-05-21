from collections import deque

class FIFOQueue:
    def __init__(self):
        """Initialize an empty queue using deque"""
        self.queue = deque()

    def push(self, item):
        """Enqueue an item (add to the back of the queue)"""
        self.queue.append(item)

    def pop(self):
        """Dequeue an item (remove from the front of the queue)"""
        if self.is_empty():
            raise IndexError("Pop from an empty queue")
        return self.queue.popleft()

    def clear(self):
        self.queue.clear()
        
    def peek(self):
        """Return the front item without removing it"""
        if self.is_empty():
            return None
        return self.queue[0]

    def is_empty(self):
        """Check if the queue is empty"""
        return len(self.queue) == 0

    def size(self):
        """Return the number of items in the queue"""
        return len(self.queue)

"""
# Example usage
fifo = FIFOQueue()
fifo.push(10)
fifo.push(20)
fifo.push(30)

print(fifo.pop())  # Output: 10 (First In, First Out)
print(fifo.peek())  # Output: 20 (Front item after pop)
print(fifo.pop())  # Output: 20
print(fifo.size())  # Output: 1
"""