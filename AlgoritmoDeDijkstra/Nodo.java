public class Nodo {
    private String id;
    private String nombre;
    private double costo;
    private Nodo anterior;

    public Nodo(String id, String nombre, double costo) {
        this.id = id;
        this.nombre = nombre;
        this.costo = costo;
        this.anterior = null;
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getNombre() {
        return nombre;
    }

    public void setNombre(String nombre) {
        this.nombre = nombre;
    }

    public double getCosto() {
        return costo;
    }

    public void setCosto(double costo) {
        this.costo = costo;
    }

    public Nodo getAnterior() {
        return anterior;
    }

    public void setAnterior(Nodo anterior) {
        this.anterior = anterior;
    }

    

}
