import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import controlP5.*; 
import processing.net.*; 
import uibooster.*; 
import uibooster.model.*; 
import uibooster.components.*; 
import java.util.Scanner; 
import processing.opengl.*; 
import guru.ttslib.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class FINAL extends PApplet {

//Imports Win1

 


 



UiBooster booster;
Client myClient;
Client myClient1;
PImage img;
PImage im3;
PImage imgb;
PImage p;
String y;
String h;
int k;
//Import Win 2

TTS tts;
PImage img10;
PImage img11;
int maxImages = 7; // Total # of images
int imageIndex = 0; // Initial image to be displayed is the first
boolean mouthOpen = false;
boolean speaking = false;
PImage[] images = new PImage[maxImages];
int mouseClicks;
String []t;

PFont f;
PFont f1;
PWindow win;
DepQ win1;
carte win2;
ControlP5 cp5;
Splashscreen splash1;


public void settings() {
     fullScreen();
}
 
public void setup() {
myClient = new Client(this, "127.0.0.1", 61915);
myClient1 = new Client(this, "127.0.0.1", 12121); 
 img =loadImage("r2d2.jpg");
p=loadImage("emineslogo2.png");
im3=loadImage("cursor.png");
img10 = loadImage("10.png");
  img11 = loadImage("8.png");
  img1 = loadImage("03.png");
  img2 = loadImage("02.png");
   f = createFont("Georgia", 40);
  f1 = createFont("Arial", 40); 
imgb =loadImage("mask.jpg");
   image(img11, 0, 0,width,height);
  cp5 = new ControlP5(this); 
   booster = new UiBooster();
  for (int i = 0; i < images.length; i ++ ) {
    images[i] = loadImage(  i+".png" );
  }
  frameRate(7);
   tts = new TTS("kevin16");
   
   
}
 
public void draw() {
   
  //if (keyPressed) {  if (key == 'b' || key == 'B') {
  //  win = new PWindow(this);
  //  noLoop();
  //  }
      
mouth();
//thread("ttsSpeak");
if (myClient.available() > 0) {
y= myClient.readString();
k=PApplet.parseInt(y);
 print(y);
if (k==0){
 mouthOpen = true;
}
else if (k==1){
   mouthOpen = false;
}
else if(k==5){
  String splashImage = dataPath("ma.jpg");
  splash1 = booster.showSplashscreen(splashImage);
  delay(1000);
  splash1.hide();}
  else if (k==6){
     String splashImage1 = dataPath("thanks.png");
    Splashscreen splash = booster.showSplashscreen(splashImage1);
    delay(5000);
    splash.hide();
    win = new PWindow(this);
  }}
  if (myClient1.available() > 0) {
win2= new carte(this);
delay(200);
 win2.exit();

  }
}
public void mousePressed() {
   if (mouseButton == LEFT) { mouseClicks++; } else { mouseClicks = 0; }
if(mouseClicks==1){
  win = new PWindow(this);
  

  //win2 = new carte(this);
  //delay(100);  
  //win2.exit();
  
//setTimeout(exit(){win2.close()};1000);
//win2.setTimeout("win2.exit()",1000);
}

else{ 
win = new PWindow(this);

}}
  public void exit()
  {
    dispose();
    win1=null;
  }



 
 
public void mouth(){
    if (mouthOpen && frameCount % 40>0 //if mouthOpen is true, mouth opens every 40 frames and stays open for 15 frames
&& frameCount % 40 <40){ 
 image(img10, 0, 0,width,height);
   image(images[imageIndex], 620, 670,700,600);
   imageIndex = PApplet.parseInt(random(images.length));}
else{ // if mouthOpen is false, mouth closes
     image(img11, 0, 0,width,height);
     
   }
}



//void ttsSpeak(){ //if speaking is true, speaking returns false, the robot speaks, and then mouthOpen returns false
//if (speaking){
//speaking = false;
//tts.speak("Hi  choose ur language");
//mouthOpen = false;
//}  
//}
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
 
  public void settings() {
      fullScreen();
 
  }
 public void setup()
 {
        
   
  image(img1, 0, 0,width,height);
 
 }


  public void draw(){
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
 
class DepQ extends PApplet {
 public boolean work = false;
  ControlP5 cpExtra1;
  PApplet parent;
   
  DepQ(PApplet app1) {
    super();
 PApplet.runSketch(new String[] {this.getClass().getSimpleName()},  this);
 
    cpExtra1 = new ControlP5(this);
 
    parent = app1;
  
        
 PFont font = createFont("Times New Roman", 50);
   cpExtra1.addButton("Deplacement")
.setPosition(1000,500)
.setSize(700,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(0xffFFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra1.addButton("Question")
.setPosition(300,500)
.setSize(500,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(0xffFFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
  }
 
  public void settings() {
      fullScreen();
  }
 


  public void draw(){
    
    background(255);
image(img, 1650, 0);
image(p, 0, 0);
 cursor(im3,13,13);
 texts();
  }
 public void texts(){
   
      
      
   text("Robot d'assistance",700,100);
        textSize(50);
        
      fill(0xff1A0099);
      }

public void Deplacement(){
   
  myClient1.write("3");
  win1.hide();
   
}// send whatever you need to send here}
  
  public void Question(){
     myClient1.write("4");
     win1.hide();
  }
  public void mouseReleased() {
  
  }
public void show() {
   work = true;
   surface.setVisible(true);
 }
 public void hide() {
   work = false;
   surface.setVisible(false);
 }}  
 
 
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
.setColorBackground(0xffFFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
cpExtra.addButton("Anglais")
.setPosition(300,500)
.setSize(500,100)
.setFont(font)
.setColorLabel(0)
.setColorBackground(0xffFFCC99)
.setColorForeground(0xff7C7C7C)
.setColorActive(0xff565656)
.getCaptionLabel()
.align(ControlP5.CENTER, ControlP5.CENTER);
  }
 
  public void settings() {
      fullScreen();
  }
 


  public void draw(){
    
    background(255);
image(img, 1650, 0);
image(p, 0, 0);
 cursor(im3,13,13);
 texts();
  }
 public void texts(){
   
      
      
   text("Robot d'assistance",700,100);
        textSize(50);
        
      fill(0xff1A0099);
      }
  
  
public void Francais(){
   
  myClient.write("1");
  
win1 = new DepQ(this);
delay(200);
win.hide1();
}// send whatever you need to send here}
   
  
  public void Anglais(){
     myClient.write("2"); 
   win1 = new DepQ(this);
delay(200);
win.hide1();}// send whatever you need to send here}}
   
  public void mouseReleased() {
  
  }
public void show1() {
   work1 = true;
   surface.setVisible(true);
 }
 public void hide1() {
   work1 = false;
   surface.setVisible(false);
 }}  
 

  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "--present", "--window-color=#666666", "--stop-color=#cccccc", "FINAL" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
