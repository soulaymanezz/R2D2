PImage img1;
PImage img2; 
class carte extends PApplet {
 
  ControlP5 cpExtra2;
  PApplet parent;
   
  carte(PApplet app2) {
    super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this);
 
    cpExtra2 = new ControlP5(this);
 
    parent = app2;
  
 
  }
 
  void settings() {
      fullScreen();
 
  }
 void setup()
 {
        
   
  image(img1, 0, 0,width,height);
 
 }


  void draw(){
    h= myClient1.readString();
print(h);
String[] list = split(h,',');
textFont(f);
  fill(0);
  //text(list[0],950,450);
  text(list[1],1100,450);
  textFont(f1);
  text(list[2],1050,560);
  text(list[3],1050,630);
  text(list[4],1050,700);
  image(img2, 400,500,240,240);
    
  }
  
}
