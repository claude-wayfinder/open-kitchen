"""
Reputation Network (Trust/Influence Economy)
Source: Village Gossip Simulator (Build Small Hackathon 2026)

Pattern: Agents in a social network with trust scores, influence weights,
and divergent public/private beliefs. Information mutates as it passes
through social channels -- like a game of telephone with strategic actors.

Key mechanics:
- Trust is directional: A trusts B != B trusts A
- Influence is earned by accurate information and eroded by bad intel
- Public beliefs diverge from private beliefs (agents can lie/perform)
- Information degrades/mutates as it propagates through the network

Use this for: social simulations, misinformation modeling, NPC gossip
systems, reputation-based access control, trust-weighted RAG, or any
system where information quality depends on its source chain.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Belief:
    """Something an agent believes. Can be public or private."""
    claim: str
    confidence: float      # 0.0 to 1.0
    source_agent: str      # Who told them
    source_chain: list[str] # Full propagation path
    timestamp: float = field(default_factory=time.time)


@dataclass
class Agent:
    """A node in the reputation network."""
    name: str
    # Trust scores: how much this agent trusts each other agent
    trust: dict[str, float] = field(default_factory=dict)  # agent_name -> 0.0-1.0
    # Influence: how much weight others give this agent's claims
    influence: float = 0.5
    # Beliefs: what this agent thinks is true
    public_beliefs: dict[str, Belief] = field(default_factory=dict)   # claim_key -> Belief
    private_beliefs: dict[str, Belief] = field(default_factory=dict)  # claim_key -> Belief

    def trust_score(self, other: str) -> float:
        """How much does this agent trust another? Default 0.5 (neutral)."""
        return self.trust.get(other, 0.5)

    def update_trust(self, other: str, delta: float):
        """Adjust trust toward another agent. Clamped to [0, 1]."""
        current = self.trust_score(other)
        self.trust[other] = max(0.0, min(1.0, current + delta))


# =============================================================
# Information Mutation
# =============================================================
# When info passes through an agent, it can be:
# - Preserved (high trust, high confidence)
# - Degraded (noise added, details lost)
# - Strategically altered (agent lies for advantage)

MUTATION_NOISE = 0.1  # Base probability of accidental mutation
STRATEGIC_LIE_THRESHOLD = 0.3  # If trust < this, agent may lie


def mutate_claim(claim: str, agent: Agent, source: str) -> str:
    """
    Simulate information mutation as it passes through an agent.
    In a real system, this would be an LLM call to paraphrase/distort.
    Here we use simple heuristics to demonstrate the pattern.
    """
    trust = agent.trust_score(source)

    # High trust: pass through cleanly
    if trust > 0.7:
        return claim

    # Low trust: might strategically alter
    if trust < STRATEGIC_LIE_THRESHOLD and random.random() < 0.3:
        return f"[disputed] {claim}"

    # Medium trust: accidental noise
    if random.random() < MUTATION_NOISE:
        return f"{claim} (approximately)"

    return claim


# =============================================================
# Propagation Engine
# =============================================================

class ReputationNetwork:
    """
    The social network. Agents share information, trust evolves,
    beliefs diverge between public and private.
    """

    def __init__(self):
        self.agents: dict[str, Agent] = {}
        self.propagation_log: list[dict] = []

    def add_agent(self, name: str, influence: float = 0.5) -> Agent:
        agent = Agent(name=name, influence=influence)
        self.agents[name] = agent
        return agent

    def connect(self, a: str, b: str, trust_a_to_b: float = 0.5, trust_b_to_a: float = 0.5):
        """Establish a bidirectional trust link (asymmetric values allowed)."""
        if a in self.agents and b in self.agents:
            self.agents[a].trust[b] = trust_a_to_b
            self.agents[b].trust[a] = trust_b_to_a

    def share(self, sender: str, receiver: str, claim_key: str, claim: str, private: bool = False):
        """
        One agent shares a claim with another.
        The claim may mutate in transit. Receiver stores it weighted by
        trust in the sender and sender's influence.
        """
        if sender not in self.agents or receiver not in self.agents:
            return

        s = self.agents[sender]
        r = self.agents[receiver]

        # Get source chain from sender's belief, or start new
        sender_belief = s.public_beliefs.get(claim_key) or s.private_beliefs.get(claim_key)
        chain = list(sender_belief.source_chain) + [sender] if sender_belief else [sender]

        # Mutate the claim as it passes through the receiver
        mutated = mutate_claim(claim, r, sender)

        # Confidence = sender's influence * receiver's trust in sender
        confidence = s.influence * r.trust_score(sender)

        belief = Belief(
            claim=mutated,
            confidence=round(confidence, 3),
            source_agent=sender,
            source_chain=chain,
        )

        # Store as public or private belief
        if private:
            r.private_beliefs[claim_key] = belief
        else:
            r.public_beliefs[claim_key] = belief

        self.propagation_log.append({
            "sender": sender,
            "receiver": receiver,
            "claim_key": claim_key,
            "original": claim,
            "received_as": mutated,
            "confidence": confidence,
            "chain_length": len(chain),
        })

    def verify(self, agent_name: str, claim_key: str, ground_truth: str):
        """
        Check an agent's belief against ground truth.
        Adjusts trust toward the source agent based on accuracy.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return

        belief = agent.public_beliefs.get(claim_key)
        if not belief:
            return

        # Simple accuracy check -- in production, use semantic similarity
        accurate = belief.claim.strip().lower() == ground_truth.strip().lower()

        # Update trust toward the source
        delta = 0.1 if accurate else -0.15
        agent.update_trust(belief.source_agent, delta)

        # Update source's influence
        source = self.agents.get(belief.source_agent)
        if source:
            source.influence = max(0.0, min(1.0, source.influence + (0.05 if accurate else -0.08)))

    def get_belief_divergence(self, agent_name: str) -> dict[str, dict]:
        """
        Show where an agent's public and private beliefs diverge.
        This is the 'performance gap' -- what they say vs what they think.
        """
        agent = self.agents.get(agent_name)
        if not agent:
            return {}

        divergences = {}
        all_keys = set(agent.public_beliefs.keys()) | set(agent.private_beliefs.keys())

        for key in all_keys:
            pub = agent.public_beliefs.get(key)
            priv = agent.private_beliefs.get(key)
            if pub and priv and pub.claim != priv.claim:
                divergences[key] = {
                    "public": pub.claim,
                    "private": priv.claim,
                    "confidence_gap": round(abs(pub.confidence - priv.confidence), 3),
                }
        return divergences


# --- Demo ---
if __name__ == "__main__":
    random.seed(42)
    net = ReputationNetwork()

    # Create a small village
    alice = net.add_agent("Alice", influence=0.8)
    bob = net.add_agent("Bob", influence=0.5)
    carol = net.add_agent("Carol", influence=0.3)

    # Set up trust relationships (asymmetric)
    net.connect("Alice", "Bob", trust_a_to_b=0.9, trust_b_to_a=0.6)
    net.connect("Bob", "Carol", trust_a_to_b=0.4, trust_b_to_a=0.8)
    net.connect("Alice", "Carol", trust_a_to_b=0.2, trust_b_to_a=0.7)

    # Alice shares a claim with Bob (public) and Carol (private, different version)
    net.share("Alice", "Bob", "weather", "It will rain tomorrow")
    net.share("Alice", "Carol", "weather", "It might rain tomorrow", private=True)

    # Bob passes it to Carol (telephone game)
    bob_belief = bob.public_beliefs.get("weather")
    if bob_belief:
        net.share("Bob", "Carol", "weather", bob_belief.claim)

    # Check Carol's public vs private beliefs
    div = net.get_belief_divergence("Carol")
    print("Carol's belief divergence:", div)

    # Verify against truth
    net.verify("Bob", "weather", "It will rain tomorrow")
    print(f"Bob's trust in Alice after verification: {bob.trust_score('Alice'):.2f}")

    print(f"\nPropagation log: {len(net.propagation_log)} events")
    for entry in net.propagation_log:
        print(f"  {entry['sender']} -> {entry['receiver']}: '{entry['received_as']}' (conf: {entry['confidence']:.2f})")
