import numpy as np

def soft_threshold(g, alpha):
    # L1 정규화(α) 적용
    if alpha == 0: 
        return g
    if g >  alpha: return g - alpha
    if g < -alpha: return g + alpha
    return 0.0

class _TreeNode:
    __slots__ = ("is_leaf","w","feat","thr","left","right","G","H")
    def __init__(self):
        self.is_leaf = True
        self.w = 0.0           # leaf weight
        self.feat = None       # split feature index
        self.thr  = None       # split threshold
        self.left = None
        self.right= None
        self.G = 0.0           # sum gradients
        self.H = 0.0           # sum hessians

class XGBoostRegressorMini:
    """
    Minimal XGBoost-like (squared loss only):
    - second-order scoring with L1/L2 regularization
    - max_depth, min_child_weight, gamma (split penalty)
    - subsample, colsample_bytree, learning_rate
    """
    def __init__(self, n_estimators=100, max_depth=3, min_child_weight=1.0,
                 reg_lambda=1.0, reg_alpha=0.0, gamma=0.0,
                 subsample=1.0, colsample_bytree=1.0,
                 learning_rate=0.1, n_candidate_splits=32, random_state=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.learning_rate = learning_rate
        self.n_candidate_splits = n_candidate_splits
        self.random_state = np.random.RandomState(random_state)
        self.trees = []
        self.feature_subsets = []

    # ---- XGBoost 핵심 수식 ----
    def _leaf_weight(self, G, H):
        # w* = - soft_threshold(G, α) / (H + λ)
        return - soft_threshold(G, self.reg_alpha) / (H + self.reg_lambda + 1e-12)

    def _score(self, G, H):
        # Obj gain of a leaf: -1/2 * (soft(G,α)^2) / (H + λ)
        g_tilde = soft_threshold(G, self.reg_alpha)
        return -0.5 * (g_tilde * g_tilde) / (H + self.reg_lambda + 1e-12)

    def _split_gain(self, G, H, GL, HL, GR, HR):
        # Gain = score(left) + score(right) - score(parent) - γ
        gain = self._score(GL, HL) + self._score(GR, HR) - self._score(G, H) - self.gamma
        return gain

    # ---- 트리 학습 ----
    def _build_tree(self, X, g, h, features, depth):
        node = _TreeNode()
        node.G = G = g.sum()
        node.H = H = h.sum()
        # leaf 조건
        if depth == self.max_depth or H < self.min_child_weight:
            node.w = self._leaf_weight(G, H)
            return node

        best_gain, best_feat, best_thr = 0.0, None, None
        GL_best = HL_best = None

        # 각 feature에서 후보 임계값을 뽑아 탐색(간단히 quantile 기반)
        for f in features:
            x = X[:, f]
            # 후보 split 후보 뽑기 (고유값이 적으면 모두 사용)
            uniq = np.unique(x)
            if len(uniq) <= 1:
                continue
            if len(uniq) > self.n_candidate_splits:
                qs = np.linspace(0.0, 1.0, self.n_candidate_splits+2)[1:-1]
                thrs = np.quantile(x, qs, method="linear")
                thrs = np.unique(thrs)
            else:
                # 인접값 중간점 사용
                thrs = (uniq[:-1] + uniq[1:]) / 2.0

            # 누적합을 빠르게 쓰기 위해 정렬
            order = np.argsort(x)
            x_sorted = x[order]
            g_sorted = g[order]
            h_sorted = h[order]
            G_prefix = np.cumsum(g_sorted)
            H_prefix = np.cumsum(h_sorted)
            G_total, H_total = G_prefix[-1], H_prefix[-1]

            # 각 threshold에 대해 왼/오 분할 집계 찾기
            for t in thrs:
                # x_sorted <= t 인 마지막 인덱스
                idx = np.searchsorted(x_sorted, t, side="right") - 1
                if idx < 0 or idx >= len(x_sorted)-1:
                    continue
                GL = G_prefix[idx]
                HL = H_prefix[idx]
                GR = G_total - GL
                HR = H_total - HL

                # child가 너무 작은 경우 방지
                if HL < self.min_child_weight or HR < self.min_child_weight:
                    continue

                gain = self._split_gain(G_total, H_total, GL, HL, GR, HR)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thr = t
                    GL_best, HL_best = GL, HL

        if best_feat is None or best_gain <= 0.0:
            node.w = self._leaf_weight(G, H)
            return node

        # 분할 확정
        node.is_leaf = False
        node.feat = best_feat
        node.thr = best_thr

        left_mask  = X[:, best_feat] <= best_thr
        right_mask = ~left_mask

        node.left  = self._build_tree(X[left_mask],  g[left_mask],  h[left_mask],  features, depth+1)
        node.right = self._build_tree(X[right_mask], g[right_mask], h[right_mask], features, depth+1)
        return node

    def _predict_tree(self, node, X):
        if node.is_leaf:
            return np.full(X.shape[0], node.w, dtype=float)
        mask = X[:, node.feat] <= node.thr
        out = np.empty(X.shape[0], dtype=float)
        out[mask]  = self._predict_tree(node.left,  X[mask])
        out[~mask] = self._predict_tree(node.right, X[~mask])
        return out

    # ---- 부스팅 루프 ----
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape

        # 초기 예측(0)에서 시작
        self.init_pred_ = np.full(n, y.mean())  # bias 초기화(편의상 평균)
        y_pred = self.init_pred_.copy()

        self.trees.clear()
        self.feature_subsets.clear()

        for m in range(self.n_estimators):
            # 2차 근사 (Squared loss: g = y_pred - y, h = 1)
            g = y_pred - y
            h = np.ones_like(g)

            # subsample
            if self.subsample < 1.0:
                idx = self.random_state.choice(n, size=int(n*self.subsample), replace=False)
            else:
                idx = np.arange(n)

            # colsample
            if self.colsample_bytree < 1.0:
                f_num = max(1, int(d * self.colsample_bytree))
                feats = self.random_state.choice(d, size=f_num, replace=False)
            else:
                feats = np.arange(d)

            tree = self._build_tree(X[idx], g[idx], h[idx], feats, depth=0)
            self.trees.append(tree)
            self.feature_subsets.append(feats)

            # 업데이트(학습률 적용)
            y_pred += self.learning_rate * self._predict_tree(tree, X)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        yhat = np.full(X.shape[0], getattr(self, "init_pred_", 0.0))
        for tree in self.trees:
            yhat += self.learning_rate * self._predict_tree(tree, X)
        return yhat
