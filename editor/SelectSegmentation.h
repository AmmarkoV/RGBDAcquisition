#ifndef SELECTSEGMENTATION_H
#define SELECTSEGMENTATION_H


#include "../acquisitionSegment/AcquisitionSegment.h"

//(*Headers(SelectSegmentation)
#include <wx/spinctrl.h>
#include <wx/checkbox.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/filedlg.h>
#include <wx/statbox.h>
#include <wx/statline.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
#include <wx/choice.h>
//*)

class SelectSegmentation: public wxDialog
{
	public:

		SelectSegmentation(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectSegmentation();

        int reloadSegmentationFormFromValues();
        int saveSegmentationValuesFromForm();

        int selectedCombinationMode;
        struct SegmentationFeaturesRGB selectedRGBConf;
        struct SegmentationFeaturesDepth selectedDepthConf;

		//(*Declarations(SelectSegmentation)
		wxSpinCtrl* minG;
		wxStaticText* StaticText24;
		wxTextCtrl* planeNormalOffset;
		wxStaticText* StaticText22;
		wxCheckBox* CheckBoxAutoPlane3Point;
		wxTextCtrl* NormalY;
		wxStaticText* StaticText21;
		wxSpinCtrl* maxR;
		wxStaticText* StaticText13;
		wxStaticText* StaticText14;
		wxStaticText* StaticText15;
		wxTextCtrl* planeP1Y;
		wxTextCtrl* cropRGBY1;
		wxTextCtrl* PlanePointX;
		wxSpinCtrl* SpinCtrlMovementG;
		wxTextCtrl* cropDepthY1;
		wxTextCtrl* cropDepthY2;
		wxSpinCtrl* eraseColorB;
		wxTextCtrl* PlanePointZ;
		wxStaticText* StaticText17;
		wxStaticText* StaticText28;
		wxTextCtrl* cropDepthX1;
		wxSpinCtrl* eraseColorG;
		wxCheckBox* CheckBoxBoundingBox;
		wxTextCtrl* cropRGBX2;
		wxTextCtrl* planeP2X;
		wxTextCtrl* bboxMinY;
		wxTextCtrl* PlanePointY;
		wxTextCtrl* planeP2Z;
		wxTextCtrl* planeP3Z;
		wxSpinCtrl* minB;
		wxSpinCtrl* minR;
		wxStaticText* StaticText20;
		wxStaticText* StaticText18;
		wxChoice* ChoiceCombination;
		wxTextCtrl* bboxMinZ;
		wxStaticText* StaticText1;
		wxStaticText* StaticText10;
		wxStaticText* StaticText16;
		wxCheckBox* CheckBoxAutoPlaneSegmentation;
		wxCheckBox* CheckBoxReplaceColor;
		wxStaticBox* StaticBox2;
		wxStaticText* StaticText3;
		wxSpinCtrl* maxG;
		wxButton* ButtonCancel;
		wxTextCtrl* planeP1X;
		wxStaticText* StaticText23;
		wxTextCtrl* cropDepthX2;
		wxTextCtrl* bboxMaxY;
		wxStaticLine* StaticLine1;
		wxTextCtrl* NormalZ;
		wxTextCtrl* cropRGBX1;
		wxButton* ButtonOk;
		wxSpinCtrl* replaceB;
		wxTextCtrl* planeP1Z;
		wxStaticText* StaticText8;
		wxStaticText* StaticText12;
		wxTextCtrl* TextCtrlMovementThreshold;
		wxButton* ButtonExport;
		wxStaticBox* StaticBox1;
		wxStaticText* StaticText7;
		wxSpinCtrl* maxDepth;
		wxTextCtrl* planeP2Y;
		wxSpinCtrl* eraseDepth;
		wxStaticText* StaticText4;
		wxCheckBox* CheckBoxSegmentRGBMovement;
		wxSpinCtrl* minDepth;
		wxCheckBox* CheckBoxSegmentMovement;
		wxStaticText* StaticText5;
		wxTextCtrl* planeNormalSize;
		wxStaticText* StaticText2;
		wxStaticText* StaticText27;
		wxTextCtrl* bboxMaxZ;
		wxSpinCtrl* SpinCtrlMovementR;
		wxCheckBox* CheckBoxPlane;
		wxStaticText* StaticText26;
		wxStaticText* StaticText6;
		wxSpinCtrl* maxB;
		wxTextCtrl* bboxMaxX;
		wxTextCtrl* NormalX;
		wxStaticText* StaticText19;
		wxTextCtrl* cropRGBY2;
		wxStaticText* StaticText29;
		wxSpinCtrl* replaceG;
		wxSpinCtrl* replaceR;
		wxSpinCtrl* SpinCtrlMovementB;
		wxSpinCtrl* eraseColorR;
		wxStaticText* StaticText9;
		wxTextCtrl* bboxMinX;
		wxTextCtrl* planeP3X;
		wxStaticText* StaticText11;
		wxStaticText* StaticText25;
		wxTextCtrl* planeP3Y;
		wxFileDialog* FileDialogExport;
		//*)

	protected:

		//(*Identifiers(SelectSegmentation)
		static const long ID_STATICBOX2;
		static const long ID_BUTTON1;
		static const long ID_BUTTON2;
		static const long ID_STATICBOX1;
		static const long ID_STATICTEXT1;
		static const long ID_SPINCTRL1;
		static const long ID_SPINCTRL2;
		static const long ID_SPINCTRL3;
		static const long ID_STATICTEXT2;
		static const long ID_SPINCTRL4;
		static const long ID_SPINCTRL5;
		static const long ID_SPINCTRL6;
		static const long ID_STATICTEXT3;
		static const long ID_STATICTEXT4;
		static const long ID_STATICTEXT5;
		static const long ID_STATICTEXT6;
		static const long ID_TEXTCTRL1;
		static const long ID_TEXTCTRL2;
		static const long ID_STATICTEXT7;
		static const long ID_TEXTCTRL3;
		static const long ID_TEXTCTRL4;
		static const long ID_STATICTEXT8;
		static const long ID_SPINCTRL7;
		static const long ID_STATICTEXT9;
		static const long ID_SPINCTRL8;
		static const long ID_STATICTEXT10;
		static const long ID_TEXTCTRL5;
		static const long ID_TEXTCTRL6;
		static const long ID_STATICTEXT11;
		static const long ID_TEXTCTRL7;
		static const long ID_TEXTCTRL8;
		static const long ID_TEXTCTRL9;
		static const long ID_TEXTCTRL10;
		static const long ID_TEXTCTRL11;
		static const long ID_TEXTCTRL12;
		static const long ID_TEXTCTRL13;
		static const long ID_TEXTCTRL14;
		static const long ID_STATICTEXT13;
		static const long ID_STATICTEXT14;
		static const long ID_STATICTEXT16;
		static const long ID_TEXTCTRL15;
		static const long ID_TEXTCTRL16;
		static const long ID_TEXTCTRL17;
		static const long ID_STATICTEXT17;
		static const long ID_TEXTCTRL18;
		static const long ID_TEXTCTRL19;
		static const long ID_TEXTCTRL20;
		static const long ID_TEXTCTRL21;
		static const long ID_TEXTCTRL22;
		static const long ID_TEXTCTRL23;
		static const long ID_STATICTEXT18;
		static const long ID_STATICTEXT19;
		static const long ID_CHOICE1;
		static const long ID_CHECKBOX1;
		static const long ID_CHECKBOX2;
		static const long ID_STATICTEXT12;
		static const long ID_SPINCTRL9;
		static const long ID_SPINCTRL10;
		static const long ID_SPINCTRL11;
		static const long ID_STATICTEXT15;
		static const long ID_SPINCTRL12;
		static const long ID_STATICLINE1;
		static const long ID_BUTTON3;
		static const long ID_CHECKBOX3;
		static const long ID_STATICTEXT20;
		static const long ID_TEXTCTRL24;
		static const long ID_CHECKBOX4;
		static const long ID_STATICTEXT21;
		static const long ID_SPINCTRL13;
		static const long ID_STATICTEXT22;
		static const long ID_SPINCTRL14;
		static const long ID_STATICTEXT23;
		static const long ID_STATICTEXT24;
		static const long ID_SPINCTRL15;
		static const long ID_STATICTEXT25;
		static const long ID_STATICTEXT26;
		static const long ID_TEXTCTRL25;
		static const long ID_CHECKBOX5;
		static const long ID_SPINCTRL16;
		static const long ID_SPINCTRL17;
		static const long ID_SPINCTRL18;
		static const long ID_CHECKBOX6;
		static const long ID_STATICTEXT27;
		static const long ID_TEXTCTRL26;
		static const long ID_TEXTCTRL27;
		static const long ID_TEXTCTRL28;
		static const long ID_TEXTCTRL29;
		static const long ID_STATICTEXT28;
		static const long ID_TEXTCTRL30;
		static const long ID_TEXTCTRL31;
		static const long ID_TEXTCTRL32;
		static const long ID_STATICTEXT29;
		static const long ID_CHECKBOX7;
		//*)

	private:

		//(*Handlers(SelectSegmentation)
		void OnButtonOkClick(wxCommandEvent& event);
		void OnButtonCancelClick(wxCommandEvent& event);
		void OnButtonExportClick(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
