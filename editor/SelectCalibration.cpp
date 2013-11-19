#include "SelectCalibration.h"

//(*InternalHeaders(SelectCalibration)
#include <wx/string.h>
#include <wx/intl.h>
//*)

//(*IdInit(SelectCalibration)
//*)

BEGIN_EVENT_TABLE(SelectCalibration,wxDialog)
	//(*EventTable(SelectCalibration)
	//*)
END_EVENT_TABLE()

SelectCalibration::SelectCalibration(wxWindow* parent,wxWindowID id)
{
	//(*Initialize(SelectCalibration)
	Create(parent, id, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxDEFAULT_DIALOG_STYLE, _T("id"));
	SetClientSize(wxSize(719,564));
	//*)
}

SelectCalibration::~SelectCalibration()
{
	//(*Destroy(SelectCalibration)
	//*)
}

