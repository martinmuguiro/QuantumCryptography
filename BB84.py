import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import Aer

def generar_bits_aleatorios(n):
    """Genera n bits aleatorios"""
    return np.random.randint(2, size=n)

def generar_bases_aleatorias(n):
    """Genera n bases aleatorias (0 para Z, 1 para X)"""
    return np.random.randint(2, size=n)

def preparar_qubit(bit, base):
    """Prepara un qubit según el bit y la base"""
    qc = QuantumCircuit(1, 1)
    
    if bit:
        qc.x(0)
    if base:
        qc.h(0)
    
    return qc

def medir_qubit(qc, base):
    """Añade medición al circuito según la base"""
    if base:
        qc.h(0)
    qc.measure(0, 0)
    return qc

def eve_intercepta(qc, eve_base):
    """Eve intercepta y mide el qubit, luego lo prepara de nuevo según su medición"""
    qr = QuantumRegister(1)
    cr_eve = ClassicalRegister(1, 'eve')
    cr_bob = ClassicalRegister(1, 'bob')
    new_qc = QuantumCircuit(qr, cr_eve, cr_bob)
    
    new_qc.compose(qc, inplace=True)
    
    if eve_base:
        new_qc.h(0)  # Si Eve elige la base X, aplica una puerta Hadamard
    new_qc.measure(0, cr_eve[0])
    
    # Después de medir, si Eve usó la base X, debe devolver el qubit a Bob con la preparación correcta
    if eve_base:
        new_qc.h(0)  # Reaplicar Hadamard si Eve usó la base X
    
    return new_qc

def bb84_con_eve(n_qubits, eve_presente=False):
    """Implementa el protocolo BB84 con opción de incluir a Eve"""
    alice_bits = generar_bits_aleatorios(n_qubits)
    alice_bases = generar_bases_aleatorias(n_qubits)
    bob_bases = generar_bases_aleatorias(n_qubits)
    eve_bases = generar_bases_aleatorias(n_qubits) if eve_presente else None
    
    backend = Aer.get_backend('aer_simulator')
    bob_results = []
    
    for i in range(n_qubits):
        qc = preparar_qubit(alice_bits[i], alice_bases[i])
        
        if eve_presente:
            qc = eve_intercepta(qc, eve_bases[i])
        
        qc = medir_qubit(qc, bob_bases[i])
        
        job = backend.run(qc, shots=1)
        result = job.result()
        counts = result.get_counts(qc)
        measured_bit = int(list(counts.keys())[0][-1])
        bob_results.append(measured_bit)
    
    mismas_bases = alice_bases == bob_bases
    clave_alice = alice_bits[mismas_bases]
    clave_bob = np.array(bob_results)[mismas_bases]
    
    n_verificar = len(clave_alice) // 4
    indices_verificacion = np.random.choice(len(clave_alice), n_verificar, replace=False)
    
    bits_verificacion_alice = clave_alice[indices_verificacion]
    bits_verificacion_bob = clave_bob[indices_verificacion]
    
    error_rate = np.sum(bits_verificacion_alice != bits_verificacion_bob) / n_verificar
    
    mascara = np.ones(len(clave_alice), dtype=bool)
    mascara[indices_verificacion] = False
    clave_final = clave_alice[mascara]
    
    return error_rate

# Función para ejecutar el experimento varias veces y calcular la media de errores
def ejecutar_experimentos(n_qubits, repeticiones=10):
    errores_sin_eve = []
    errores_con_eve = []
    
    # Ejecutar sin Eve
    for _ in range(repeticiones):
        error_sin_eve = bb84_con_eve(n_qubits, eve_presente=False)
        errores_sin_eve.append(error_sin_eve)
    
    # Ejecutar con Eve
    for i in range(repeticiones):
        error_con_eve = bb84_con_eve(n_qubits, eve_presente=True)
        errores_con_eve.append(error_con_eve)
        print(f"Repetición {i+1} con Eve: Error de verificación = {error_con_eve * 100:.2f}%")
    
    # Calcular la media de errores
    media_error_sin_eve = np.mean(errores_sin_eve)
    media_error_con_eve = np.mean(errores_con_eve)
    
    return media_error_sin_eve, media_error_con_eve

# Ejemplo de ejecución
if __name__ == "__main__":
    n_qubits = 1000  # Número de qubits en cada ejecución
    repeticiones = 10  # Número de repeticiones
    
    media_error_sin_eve, media_error_con_eve = ejecutar_experimentos(n_qubits, repeticiones)
    
    print(f"\nMedia de error sin Eve: {media_error_sin_eve * 100:.2f}%")
    print(f"Media de error con Eve: {media_error_con_eve * 100:.2f}%")
