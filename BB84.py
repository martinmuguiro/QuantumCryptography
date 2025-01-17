import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

def generate_random_bits(n):
    """Generates n random bits"""
    return np.random.randint(2, size=n)

def generate_random_bases(n):
    """Generates n random bases (0 for Z, 1 for X)"""
    return np.random.randint(2, size=n)

def prepare_qubit(bit, basis):
    """Prepares a qubit according to bit and basis"""
    qc = QuantumCircuit(1, 1)
    
    if bit:
        qc.x(0)
    if basis:
        qc.h(0)
    
    return qc

def measure_qubit(qc, basis):
    """Adds measurement to circuit according to basis"""
    if basis:
        qc.h(0)
    qc.measure(0, 0)
    return qc

def eve_intercepts(qc, eve_basis):
    """Eve intercepts and measures the qubit, then prepares it again according to her measurement"""
    qr = QuantumRegister(1)
    cr_eve = ClassicalRegister(1, 'eve')
    cr_bob = ClassicalRegister(1, 'bob')
    new_qc = QuantumCircuit(qr, cr_eve, cr_bob)
    
    new_qc.compose(qc, inplace=True)
    
    if eve_basis:
        new_qc.h(0)  # If Eve chooses X basis, applies Hadamard gate
    new_qc.measure(0, cr_eve[0])
    
    # After measuring, if Eve used X basis, must return qubit to Bob with correct preparation
    if eve_basis:
        new_qc.h(0)  # Reapply Hadamard if Eve used X basis
    
    return new_qc

def bb84_with_eve(n_qubits, eve_present=False):
    """Implements BB84 protocol with option to include Eve"""
    alice_bits = generate_random_bits(n_qubits)
    alice_bases = generate_random_bases(n_qubits)
    bob_bases = generate_random_bases(n_qubits)
    eve_bases = generate_random_bases(n_qubits) if eve_present else None
    
    backend = Aer.get_backend('aer_simulator')
    bob_results = []
    
    for i in range(n_qubits):
        qc = prepare_qubit(alice_bits[i], alice_bases[i])
        
        if eve_present:
            qc = eve_intercepts(qc, eve_bases[i])
        
        qc = measure_qubit(qc, bob_bases[i])
        
        job = backend.run(qc, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        measured_bit = int(list(counts.keys())[0][-1])
        bob_results.append(measured_bit)
    
    # Calculate matching bases
    same_bases = alice_bases == bob_bases
    matching_bases_count = np.sum(same_bases)
    
    alice_key = alice_bits[same_bases]
    bob_key = np.array(bob_results)[same_bases]
    
    n_verify = len(alice_key) // 4
    verification_indices = np.random.choice(len(alice_key), n_verify, replace=False)
    
    verification_bits_alice = alice_key[verification_indices]
    verification_bits_bob = bob_key[verification_indices]
    
    error_rate = np.sum(verification_bits_alice != verification_bits_bob) / n_verify
    
    mask = np.ones(len(alice_key), dtype=bool)
    mask[verification_indices] = False
    final_key = alice_key[mask]
    
    # Return error rate and matching bases count
    return error_rate, matching_bases_count / n_qubits  

def run_experiments(n_qubits, repetitions=10):
    """Runs the experiment multiple times and calculates average errors and matching bases"""
    errors_without_eve = []
    errors_with_eve = []
    matching_bases_without_eve = []
    matching_bases_with_eve = []
    
    for _ in range(repetitions):
        error_without_eve, matching_bases_without_eve_exp = bb84_with_eve(n_qubits, eve_present=False)
        errors_without_eve.append(error_without_eve)
        matching_bases_without_eve.append(matching_bases_without_eve_exp)
    
    for _ in range(repetitions):
        error_with_eve, matching_bases_with_eve_exp = bb84_with_eve(n_qubits, eve_present=True)
        errors_with_eve.append(error_with_eve)
        matching_bases_with_eve.append(matching_bases_with_eve_exp)
    
    mean_error_without_eve = np.mean(errors_without_eve)
    mean_error_with_eve = np.mean(errors_with_eve)
    mean_matching_bases_without_eve = np.mean(matching_bases_without_eve)
    mean_matching_bases_with_eve = np.mean(matching_bases_with_eve)
    
    return mean_error_without_eve, mean_error_with_eve, mean_matching_bases_without_eve, mean_matching_bases_with_eve


if __name__ == "__main__":
    n_qubits = 1000  # Amount of qubits 
    repetitions = 100  # Amount of repetitions
    
    mean_error_without_eve, mean_error_with_eve, mean_matching_bases_without_eve, mean_matching_bases_with_eve = run_experiments(n_qubits, repetitions)
    
    print(f"\nMean error with Eve: {mean_error_with_eve * 100:.2f}%")
    print(f"Mean error without Eve: {mean_error_without_eve * 100:.2f}%")
    print(f"Mean matching bases with Eve: {mean_matching_bases_with_eve * 100:.2f}%")
    print(f"Mean matching bases without Eve: {mean_matching_bases_without_eve * 100:.2f}%")
