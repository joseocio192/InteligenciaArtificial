public class main {
    public static void main(String[] args) {
        Arbol arbol = new Arbol();
        int[] valores = {10, 5, 15, 3, 7, 11};
        for (int valor : valores) {
            arbol.insertar(valor);
        }
        arbol.ImprimirArbol();
        System.out.println();
        Nodo nodoBuscado = arbol.buscarNodo(5);
        if (nodoBuscado != null) {
            System.out.println("Nodo encontrado: " + nodoBuscado.getValor());
        } else {    
            System.out.println("Nodo no encontrado");
        }
    }
}
