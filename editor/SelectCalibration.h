#ifndef SELECTCALIBRATION_H
#define SELECTCALIBRATION_H

//(*Headers(SelectCalibration)
#include <wx/dialog.h>
//*)

class SelectCalibration: public wxDialog
{
	public:

		SelectCalibration(wxWindow* parent,wxWindowID id=wxID_ANY);
		virtual ~SelectCalibration();

		//(*Declarations(SelectCalibration)
		//*)

	protected:

		//(*Identifiers(SelectCalibration)
		//*)

	private:

		//(*Handlers(SelectCalibration)
		//*)

		DECLARE_EVENT_TABLE()
};

#endif
