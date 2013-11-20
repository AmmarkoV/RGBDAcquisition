#ifndef SELECTCALIBRATION_H
#define SELECTCALIBRATION_H


#include "../tools/Calibration/calibration.h"

//(*Headers(SelectCalibration)
#include <wx/combobox.h>
#include <wx/dialog.h>
#include <wx/button.h>
#include <wx/statbox.h>
#include <wx/stattext.h>
#include <wx/textctrl.h>
//*)

class SelectCalibration: public wxDialog
{
	public:

        struct calibration calib;

		SelectCalibration(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectCalibration();


        int reloadCalibrationFormFromValues();
        int saveCalibrationValuesFromForm();

		//(*Declarations(SelectCalibration)
		wxTextCtrl* camera3;
		wxTextCtrl* k3;
		wxStaticText* StaticText21;
		wxStaticText* StaticText14;
		wxStaticText* StaticText15;
		wxStaticText* fyLabel;
		wxTextCtrl* height;
		wxStaticText* StaticText17;
		wxTextCtrl* width;
		wxTextCtrl* farPlane;
		wxTextCtrl* rY;
		wxStaticText* cxLabel;
		wxTextCtrl* p2;
		wxTextCtrl* rX;
		wxTextCtrl* camera7;
		wxTextCtrl* nearPlane;
		wxTextCtrl* rZ;
		wxStaticText* StaticText20;
		wxStaticText* StaticText18;
		wxStaticText* fxLabel;
		wxStaticText* StaticText10;
		wxStaticText* StaticText16;
		wxStaticBox* StaticBox2;
		wxTextCtrl* camera1;
		wxStaticText* StaticText3;
		wxButton* ButtonCancel;
		wxButton* ButtonOk;
		wxTextCtrl* camera5;
		wxStaticText* StaticText8;
		wxStaticBox* StaticBox1;
		wxStaticText* StaticText7;
		wxStaticBox* StaticBox3;
		wxTextCtrl* camera4;
		wxTextCtrl* tX;
		wxStaticText* StaticText4;
		wxTextCtrl* camera6;
		wxTextCtrl* p1;
		wxStaticText* StaticText5;
		wxStaticText* StaticText2;
		wxTextCtrl* tZ;
		wxTextCtrl* k1;
		wxTextCtrl* camera8;
		wxComboBox* ComboBoxScale;
		wxTextCtrl* k2;
		wxStaticText* StaticText6;
		wxTextCtrl* camera2;
		wxStaticText* StaticText19;
		wxTextCtrl* camera0;
		wxStaticText* StaticText9;
		wxTextCtrl* tY;
		wxStaticText* cyLabel;
		//*)

	protected:

		//(*Identifiers(SelectCalibration)
		static const long ID_STATICBOX1;
		static const long ID_STATICTEXT1;
		static const long ID_TEXTCTRL1;
		static const long ID_TEXTCTRL2;
		static const long ID_TEXTCTRL3;
		static const long ID_TEXTCTRL4;
		static const long ID_TEXTCTRL5;
		static const long ID_TEXTCTRL6;
		static const long ID_TEXTCTRL7;
		static const long ID_TEXTCTRL8;
		static const long ID_TEXTCTRL9;
		static const long ID_STATICTEXT2;
		static const long ID_TEXTCTRL10;
		static const long ID_TEXTCTRL11;
		static const long ID_TEXTCTRL12;
		static const long ID_TEXTCTRL13;
		static const long ID_TEXTCTRL14;
		static const long ID_STATICTEXT3;
		static const long ID_STATICTEXT4;
		static const long ID_STATICTEXT5;
		static const long ID_STATICTEXT6;
		static const long ID_STATICTEXT7;
		static const long ID_STATICTEXT8;
		static const long ID_STATICBOX2;
		static const long ID_STATICTEXT9;
		static const long ID_TEXTCTRL15;
		static const long ID_TEXTCTRL16;
		static const long ID_TEXTCTRL17;
		static const long ID_STATICTEXT10;
		static const long ID_TEXTCTRL18;
		static const long ID_TEXTCTRL19;
		static const long ID_TEXTCTRL20;
		static const long ID_STATICTEXT11;
		static const long ID_STATICTEXT12;
		static const long ID_STATICTEXT13;
		static const long ID_STATICTEXT14;
		static const long ID_STATICTEXT15;
		static const long ID_STATICTEXT16;
		static const long ID_STATICTEXT17;
		static const long ID_STATICTEXT18;
		static const long ID_STATICTEXT19;
		static const long ID_COMBOBOX1;
		static const long ID_BUTTON1;
		static const long ID_BUTTON2;
		static const long ID_STATICBOX3;
		static const long ID_STATICTEXT20;
		static const long ID_TEXTCTRL21;
		static const long ID_TEXTCTRL22;
		static const long ID_STATICTEXT21;
		static const long ID_TEXTCTRL23;
		static const long ID_TEXTCTRL24;
		//*)

	private:

		//(*Handlers(SelectCalibration)
		void OnButtonOkClick(wxCommandEvent& event);
		void OnButtonCancelClick(wxCommandEvent& event);
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
