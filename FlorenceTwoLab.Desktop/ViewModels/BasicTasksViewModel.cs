using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using FlorenceTwoLab.Core;

namespace FlorenceTwoLab.Desktop.ViewModels;

public partial class BasicTaskGroupViewModel : ObservableObject, ITaskGroupViewModel
{
    [ObservableProperty] private IReadOnlyCollection<Florence2TaskType> _predefinedTasks =
    [
        Florence2TaskType.Caption,
        Florence2TaskType.DetailedCaption,
        Florence2TaskType.MoreDetailedCaption,
        Florence2TaskType.Ocr,
        Florence2TaskType.OcrWithRegions,
        Florence2TaskType.ObjectDetection,
        Florence2TaskType.DenseRegionCaption,
        Florence2TaskType.RegionProposal
    ];

    [ObservableProperty] private Florence2TaskType _selectedTask;
    
    private Action<Florence2TaskType>? _runTask;

    public BasicTaskGroupViewModel()
    {
        SelectedTask = _predefinedTasks.First();
    }
    
    public string Header => "Basic";

    public Florence2Query Query() => Florence2Tasks.CreateQuery(SelectedTask);
    
    public void SelectFirstTask()
    {
        // we can not rely on the change event to trigger the task
        if (SelectedTask == PredefinedTasks.First()) _runTask?.Invoke(SelectedTask);
        SelectedTask = PredefinedTasks.First();
    }

    public ITaskGroupViewModel Initialize(Action<Florence2TaskType> runTask)
    {
        _runTask = runTask;
        return this;
    }
    
    partial void OnSelectedTaskChanged(Florence2TaskType value) => _runTask?.Invoke(value);
}