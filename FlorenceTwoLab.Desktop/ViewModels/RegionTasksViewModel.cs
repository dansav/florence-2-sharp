using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using FlorenceTwoLab.Core;

namespace FlorenceTwoLab.Desktop.ViewModels;

public partial class RegionTaskGroupViewModel : ObservableObject, ITaskGroupViewModel
{
    private readonly IImageSelectionSource _imageSelectionSource;

    [ObservableProperty] private IReadOnlyCollection<Florence2TaskType> _predefinedTasks =
    [
        Florence2TaskType.RegionToSegmentation,
        Florence2TaskType.RegionToCategory,
        Florence2TaskType.RegionToDescription,
        Florence2TaskType.RegionToOcr
    ];

    [ObservableProperty] private Florence2TaskType _selectedTask;

    private Action<Florence2TaskType>? _runTask;

    public RegionTaskGroupViewModel(IImageSelectionSource imageSelectionSource)
    {
        _imageSelectionSource = imageSelectionSource;
        SelectedTask = _predefinedTasks.First();
    }
    
    public string Header => "Region";

    public ITaskGroupViewModel Initialize(Action<Florence2TaskType> runTask)
    {
        _runTask = runTask;
        return this;
    }

    public Florence2Query Query() =>
        Florence2Tasks.CreateQuery(SelectedTask, _imageSelectionSource.Selection, _imageSelectionSource.ImageSize);

    public void SelectFirstTask()
    {
        // we can not rely on the change event to trigger the task
        if (SelectedTask == PredefinedTasks.First()) _runTask?.Invoke(SelectedTask);
        SelectedTask = PredefinedTasks.First();
    }

    partial void OnSelectedTaskChanged(Florence2TaskType value) => _runTask?.Invoke(value);

    [RelayCommand]
    private void Create()
    {
        Debug.WriteLine("CREATE REGION");
    }
}