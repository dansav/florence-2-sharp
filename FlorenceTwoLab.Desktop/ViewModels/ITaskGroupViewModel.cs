using System;
using FlorenceTwoLab.Core;

namespace FlorenceTwoLab.Desktop.ViewModels;

public interface ITaskGroupViewModel
{
    string Header { get; }
    
    ITaskGroupViewModel Initialize(Action<Florence2TaskType> runTask);
    
    void SelectFirstTask();

    Florence2Query Query();
}