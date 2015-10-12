use <MCAD/fonts.scad>


oX = 0.3;
oY = 0.7; 
thickness=0.6; 

//Our holes
   DX1=1.0; DY1=0.5;
   DX2=5.5; DY2=1.3;
   DX3=3.5; DY3=1.5;
   DX4=1.5; DY4=2.5;
   DX5=6.5; DY5=2.5;

labXOff=0.4;
labYOff=0.9;

thisFont=8bit_polyfont();
x_shift=thisFont[0][0];
y_shift=thisFont[0][1];

labelStrings=["dummy","D1","D2","D3","D4","D5","Made By Ammar","D1-D2=4.57cm","D1-D5=5.85cm","D4-D2=4.17cm","D4-D5=5cm"];
 


module drawLabel(posX,posY,i,scale)
{ 
 translate([posX, posY , 0]) 
 { 
  rotate ( [180,0,0] )
  {
   scale([scale,scale,0.15 ])
   {
   assign( theseIndicies=search(labelStrings[i],thisFont[2],1,1) ) 
    for( j=[0:(len(theseIndicies)-1)] ) translate([j*x_shift,-y_shift/2]) 
      {
       linear_extrude(height=thickness) 
       polygon(points=thisFont[2][theseIndicies[j]][6][0],paths=thisFont[2][theseIndicies[j]][6][1]);
      }
    }
  }
 } 
}



module remove6mmBolt(posX,posY)
  { 
   $fn=100;   
    translate([posX,posY,0]) 
    {  
      cylinder(h=thickness+1,r=0.302);  
    }   
  }
 



module apostatis()
{ 
   $fn=100; 
     
   difference()
   {      
      union() 
       {
		  color([255,0,0,255])
        {
         translate([1.9,0,0]) 
         { 
           cube([oX+4.1/* cm width */  ,  oY+3.7/* cm height */, thickness /* cm thick*/]); }  
         }


		  color([255,0,0,255])
        { 
         translate([oX+DX1,oY+DY1,0])  { cylinder(h=thickness,r=1.30);  }
         translate([oX+DX2,oY+DY2,0])  { cylinder(h=thickness,r=1.30);  }
         translate([oX+DX3,oY+DY3,0])  { cylinder(h=thickness,r=1.30);  }
         translate([oX+DX4,oY+DY4,0])  { cylinder(h=thickness,r=1.30);  }
         translate([oX+DX5,oY+DY5,0])  { cylinder(h=thickness,r=1.30);  }
        }
         
		  color([255,255,0,255])
        { 
         drawLabel(oX+DX1-labXOff,oY+DY1-labYOff,1,0.05);
         drawLabel(oX+DX2-labXOff,oY+DY2-labYOff,2,0.05);
         drawLabel(oX+DX3-labXOff,oY+DY3-labYOff,3,0.05);
         drawLabel(oX+DX4-labXOff,oY+DY4-labYOff,4,0.05);
         drawLabel(oX+DX5-labXOff,oY+DY5-labYOff,5,0.05);

         drawLabel(oX+DX5-labXOff-0.35,oY+DY5+labYOff,6,0.015);
        }

		  color([0,0,255,255])
        { 
         drawLabel(oX+DX1+1,oY+DY1-labYOff,7,0.03);
         drawLabel(oX+DX1+1,oY+DY1-labYOff+0.4 ,8,0.03);

         drawLabel(oX+DX4+1,oY+DY4-labYOff+0.8,7,0.03);
         drawLabel(oX+DX4+1,oY+DY4-labYOff+0.8+0.4 ,8,0.03);
        }

       } 
         
        union() 
       { 
         remove6mmBolt(oX+DX1,oY+DY1);
         remove6mmBolt(oX+DX2,oY+DY2);
         remove6mmBolt(oX+DX3,oY+DY3);
         remove6mmBolt(oX+DX4,oY+DY4);
         remove6mmBolt(oX+DX5,oY+DY5); 
       }  
 
   } 

}

allLabels();
apostatis();
