def player(prev_play, opponent_history=[], my_history=[], data={}):
    """
    Multi-strategy RPS player that adapts to opponent patterns.

    Strategy selection: each model predicts the opponent's next move.
    We score each model on historical accuracy and use the best one.
    Special handling: abbey-style bots are countered by simulating their
    internal state directly.
    """
    # BEATS[x] = the move that beats x  (P beats R, S beats P, R beats S)
    BEATS = {"R": "P", "P": "S", "S": "R"}

    # ── First call of a new match ─────────────────────────────────────────
    if prev_play == "":
        opponent_history.clear()
        my_history.clear()
        data.clear()
        data["scores"] = {}
        data["last_preds"] = {}

    if prev_play:
        opponent_history.append(prev_play)

    n = len(opponent_history)

    # ── Score last round's predictions ───────────────────────────────────
    if prev_play and data.get("last_preds"):
        for k, pred in data["last_preds"].items():
            hit = 1 if pred == prev_play else -1
            data["scores"][k] = data["scores"].get(k, 0) + hit

    # ── Model definitions ─────────────────────────────────────────────────
    # Each model returns its prediction of the opponent's NEXT move.

    def m_quincy():
        """Quincy cycles R R P P S regardless of our play."""
        return ["R", "R", "P", "P", "S"][n % 5]

    def m_repeat():
        """Opponent repeats their last move."""
        return opponent_history[-1] if n else "R"

    def m_cycle_up():
        """Opponent cycles up: R→P→S→R"""
        if not n: return "R"
        idx = "RPS".index(opponent_history[-1])
        return "RPS"[(idx + 1) % 3]

    def m_cycle_down():
        """Opponent cycles down: R→S→P→R"""
        if not n: return "R"
        idx = "RPS".index(opponent_history[-1])
        return "RPS"[(idx - 1) % 3]

    def m_markov1():
        """1-gram Markov on opponent history."""
        if n < 2: return "R"
        last = opponent_history[-1]
        counts = {"R": 0, "P": 0, "S": 0}
        for i in range(n - 1):
            if opponent_history[i] == last:
                counts[opponent_history[i + 1]] += 1
        return max(counts, key=counts.get)

    def m_markov2():
        """2-gram Markov on opponent history."""
        if n < 3: return "R"
        pair = (opponent_history[-2], opponent_history[-1])
        counts = {"R": 0, "P": 0, "S": 0}
        for i in range(n - 2):
            if (opponent_history[i], opponent_history[i+1]) == pair:
                counts[opponent_history[i + 2]] += 1
        return max(counts, key=counts.get)

    def m_kris():
        """
        Kris always plays BEATS[our_last_move] (counters our last move).
        So predict BEATS[my_last].
        """
        if not my_history: return "R"
        return BEATS[my_history[-1]]

    def m_mrugesh():
        """
        Mrugesh plays BEATS[most_common_of_our_last_10].
        Predict that.
        """
        if not my_history: return "R"
        window = my_history[-10:]
        counts = {"R": 0, "P": 0, "S": 0}
        for m in window: counts[m] += 1
        most_common = max(counts, key=counts.get)
        return BEATS[most_common]

    def m_abbey():
        """
        Abbey builds a 2-gram Markov on OUR moves and plays BEATS[predicted_our].
        We simulate abbey's internal state to predict her move exactly.
        """
        h = my_history  # abbey sees OUR moves as her "opponent_history"
        hn = len(h)
        if hn < 2: return "R"
        last_two = h[-2] + h[-1]
        counts = {"R": 0, "P": 0, "S": 0}
        for i in range(hn - 2):
            pair = h[i] + h[i + 1]
            if pair == last_two and i + 2 < hn:
                counts[h[i + 2]] += 1
        if max(counts.values()) == 0:
            return "R"
        abbey_pred_our = max(counts, key=counts.get)
        # abbey plays what beats her predicted version of our move
        return BEATS[abbey_pred_our]

    models = {
        "quincy":   m_quincy,
        "repeat":   m_repeat,
        "cycle_up": m_cycle_up,
        "cycle_dn": m_cycle_down,
        "mk1":      m_markov1,
        "mk2":      m_markov2,
        "kris":     m_kris,
        "mrugesh":  m_mrugesh,
        "abbey":    m_abbey,
    }

    if "scores" not in data:
        data["scores"] = {k: 0 for k in models}
    else:
        for k in models:
            if k not in data["scores"]:
                data["scores"][k] = 0

    # ── Generate predictions ──────────────────────────────────────────────
    preds = {}
    for k, fn in models.items():
        try:
            preds[k] = fn()
        except Exception:
            preds[k] = "R"

    data["last_preds"] = preds

    # ── Pick the best model ───────────────────────────────────────────────
    if n < 5:
        # Warm-up: vote across all models
        votes = {"R": 0, "P": 0, "S": 0}
        for pred in preds.values():
            votes[pred] += 1
        opp_pred = max(votes, key=votes.get)
    else:
        best = max(data["scores"], key=lambda k: data["scores"][k])
        opp_pred = preds[best]

    my_move = BEATS[opp_pred]
    my_history.append(my_move)
    return my_move
