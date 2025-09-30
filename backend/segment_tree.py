class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.build(data, 0, 0, self.n - 1)


    """The time complexity for this function is O(n) 
    since each leaf node is visited exactly once and , a complete segment tree has 
    at most 4n nodes for an array of size n """

    def build(self, data, node, l, r):
        if l == r:
            self.tree[node] = (data[l], data[l], data[l], 1)  # (min, max, sum, count)
            return
        mid = (l + r) // 2
        self.build(data, 2*node+1, l, mid)
        self.build(data, 2*node+2, mid+1, r)
        self.tree[node] = self._merge(self.tree[2*node+1], self.tree[2*node+2])


        """The time complexity for this function is O(1) since its called once
    per internal node during the call"""

    def _merge(self, left, right):
        return (
            min(left[0], right[0]),
            max(left[1], right[1]),
            left[2] + right[2],
            left[3] + right[3]
        )

    """Range query works in O(log n), since at most log(n) segments are merged."""

    def query(self, ql, qr, node=0, l=0, r=None):
        if r is None:
            r = self.n - 1
        if qr < l or ql > r:
            return (float('inf'), float('-inf'), 0, 0)
        if ql <= l and r <= qr:
            return self.tree[node]
        mid = (l + r) // 2
        left = self.query(ql, qr, 2*node+1, l, mid)
        right = self.query(ql, qr, 2*node+2, mid+1, r)
        return self._merge(left, right)

    def range_min(self, l, r): 
        return self.query(l, r)[0]
    
    def range_max(self, l, r): 
        return self.query(l, r)[1]
    
    def range_sum(self, l, r): 
        return self.query(l, r)[2]
    
    def range_avg(self, l, r):
        q = self.query(l, r)
        return q[2]/q[3] if q[3] else None
    

    """This function traverses the tree from the root to the leaf and back,
      a path of length O(logn), performing constant work at each step."""
    
    def update(self, index, value, node=0, l=0, r=None):
        if r is None:
            r = self.n - 1
        
        self._propagate_lazy(node, l, r) # Push down any pending lazy updates
        
        if l == r:
            self.data[index] = value
            self.tree[node] = (value, value, value, 1)
            return

        mid = (l + r) // 2
        if l <= index <= mid:
            self.update(index, value, 2*node+1, l, mid)
        else:
            self.update(index, value, 2*node+2, mid+1, r)
        
        self.tree[node] = self._merge(self.tree[2*node+1], self.tree[2*node+2])


        """ O(logn). It uses lazy propagation to update segments within the range, 
    only visiting a logarithmic number of nodes instead of every node in the range."""

        
    def range_update(self, ql, qr, value, node=0, l=0, r=None):
        if r is None:
            r = self.n - 1

        self._propagate_lazy(node, l, r)

        if l > r or l > qr or r < ql:
            return
        
        if ql <= l and r <= qr:
            self.tree[node] = (value, value, (r - l + 1) * value, (r - l + 1))
            if l != r:
                self.lazy[2 * node + 1] = value
                self.lazy[2 * node + 2] = value
            return
        
        mid = (l + r) // 2
        self.range_update(ql, qr, value, 2 * node + 1, l, mid)
        self.range_update(ql, qr, value, 2 * node + 2, mid + 1, r)

        self.tree[node] = self._merge(self.tree[2 * node + 1], self.tree[2 * node + 2])


    """ O(1).This helper function performs a fixed number of operations to push a 
    pending update down to children nodes, without any recursive calls."""

    def _propagate_lazy(self, node, l, r):
        if self.lazy[node] != 0:
            value = self.lazy[node]
            # Update current node with lazy value
            self.tree[node] = (value, value, (r-l+1) * value, (r-l+1))
            
            # If not a leaf node, push lazy value to children
            if l != r:
                self.lazy[2*node+1] = value
                self.lazy[2*node+2] = value
            
            self.lazy[node] = 0 # Reset lazy value for current node