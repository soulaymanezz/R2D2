 
class PWindow extends PApplet {
  public boolean work1 = false;
  ControlP5 cpExtra;
  PApplet parent;
   
  PWindow(PApplet app) {
    super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this);
 
    cpExtra = new ControlP5(this);
 
    parent = app;
  
        
 PFont font = createFont("Times New Roman", 50);
   cpExtra.addButton("Francais")
.setPosition(1000,500)
.setSize(700,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(#FFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra.addButton("Anglais")
.setPosition(300,500)
.setSize(500,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(#FFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
  }
 
  void settings() {
      fullScreen();
  }
 


  void draw(){
    
    background(255);
image(img, 1650, 0);
image(p, 0, 0);
 cursor(im3,13,13);
 texts();
  }
 void texts(){
   
      
      
   text("Robot d'assistance",700,100);
        textSize(50);
        
      fill(#1A0099);
      }
  
  
void Francais(){
   
  myClient.write("1");
  
win1 = new DepQ(this);
delay(200);
win.hide1();
}// send whatever you need to send here}
   
  
  void Anglais(){
     myClient.write("2"); 
   win1 = new DepQ(this);
delay(200);
win.hide1();}// send whatever you need to send here}}
   
  void mouseReleased() {
  
  }
public void show1() {
   work1 = true;
   surface.setVisible(true);
 }
 public void hide1() {
   work1 = false;
   surface.setVisible(false);
 }}  
 
