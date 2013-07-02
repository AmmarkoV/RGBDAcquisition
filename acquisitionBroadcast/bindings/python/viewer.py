# draw lines, a rounded-rectangle and a circle on a wx.PaintDC() surface
# tested with Python24 and wxPython26     vegaseat      06mar2007
# Works with Python2.5 on OSX bcl 28Nov2008

import wx 
import httplib2

class MyFrame(wx.Frame): 
    """a frame with a panel"""
    def __init__(self, parent=None, id=-1, title=None): 
        wx.Frame.__init__(self, parent, id, title) 
        self.panel = wx.Panel(self, size=(640, 480)) 
        self.panel.Bind(wx.EVT_PAINT, self.on_paint) 
        self.Fit() 

    def on_paint(self, event):
        resp, rgb = httplib2.Http().request("http://139.91.185.49:8082/rgb.raw")
        resp, depth = httplib2.Http().request("http://139.91.185.49:8082/depth.raw")
        # establish the painting surface
        dc = wx.PaintDC(self.panel)

        rgbBitmap = wx.BitmapFromBits (rgb, 640, 480, 24) 
        dc.DrawBitmap( rgbBitmap, 1 , 1) 
        depthBitmap = wx.BitmapFromBits (depth, 640, 480, 16) 
        dc.DrawBitmap( depthBitmap, 600 , 1 , 1) 



        #dc.SetPen(wx.Pen('blue', 4))
        # draw a blue line (thickness = 4)
        #dc.DrawLine(50, 20, 300, 20)
        #dc.SetPen(wx.Pen('red', 1))
        # draw a red rounded-rectangle
        #rect = wx.Rect(50, 50, 100, 100) 
        #dc.DrawRoundedRectangleRect(rect, 8)
        # draw a red circle with yellow fill
        #dc.SetBrush(wx.Brush('yellow'))
        #x = 250
        #y = 100
        #r = 50
        #dc.DrawCircle(x, y, r) 
         

# test it ...
app = wx.PySimpleApp() 
frame1 = MyFrame(title='Network Viewer') 
frame1.Center() 
frame1.Show() 
app.MainLoop()
