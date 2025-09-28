# segment_tree.py
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.build(data, 0, 0, self.n - 1)

    def build(self, data, node, l, r):
        if l == r:
            self.tree[node] = (data[l], data[l], data[l], 1)  # (min, max, sum, count)
            return
        mid = (l + r) // 2
        self.build(data, 2*node+1, l, mid)
        self.build(data, 2*node+2, mid+1, r)
        self.tree[node] = self._merge(self.tree[2*node+1], self.tree[2*node+2])

    def _merge(self, left, right):
        return (
            min(left[0], right[0]),
            max(left[1], right[1]),
            left[2] + right[2],
            left[3] + right[3]
        )

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

    def range_min(self, l, r): return self.query(l, r)[0]
    def range_max(self, l, r): return self.query(l, r)[1]
    def range_sum(self, l, r): return self.query(l, r)[2]
    def range_avg(self, l, r):
        q = self.query(l, r)
        return q[2]/q[3] if q[3] else None
