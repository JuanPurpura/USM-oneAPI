//==============================================================
// Copyright © Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#include <sycl/sycl.hpp> // Incluye la cabecera de SYCL
using namespace sycl; // Simplifica el uso de los nombres de espacio SYCL

static const int N = 16; // Declara una constante N con valor 16

int main() {
  // Crea una cola de ejecución SYCL
  queue q; 
  
  // Imprime el nombre del dispositivo en el que se está ejecutando el código
  std::cout << "Dispositivo: " << q.get_device().get_info<info::device::name>() << "\n"; 

  //# Asignación de USM utilizando malloc_shared
  // Asigna memoria compartida utilizando USM y lo asocia con la cola de ejecución q. El puntero data apunta a esta memoria asignada.
  int *data = malloc_shared<int>(N, q);

  //# Inicialización del array de datos
  // Inicializa el array data con valores del 0 al 15 mediante un bucle for
  for (int i = 0; i < N; i++) data[i] = i; 

  //# Modificación del array de datos en el dispositivo
  // Modifica el array data multiplicando cada elemento por 2 en paralelo utilizando un kernel SYCL
  q.parallel_for(range<1>(N), [=](id<1> i) { data[i] *= 2; }).wait(); 

  //# Impresión de la salida
  // Imprime los valores del array data después de la modificación en el kernel
  for (int i = 0; i < N; i++) std::cout << data[i] << "\n"; 
    
  // Libera la memoria compartida
  free(data, q); 
    
  return 0; // Retorna 0 para indicar que el programa ha finalizado correctamente
}