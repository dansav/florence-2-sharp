using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

using FlorenceTwoLab.Core;
using FlorenceTwoLab.Desktop.Models;

using SixLabors.ImageSharp;

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
    private Rectangle? _region;
    private Size? _imageSize;

    public RegionTaskGroupViewModel(IImageSelectionSource imageSelectionSource)
    {
        _imageSelectionSource = imageSelectionSource;
        _imageSelectionSource.ImageSelectionChanged += ImageSelectionSourceOnImageSelectionChanged;
        SelectedTask = _predefinedTasks.First();
    }

    public string Header => "Region";

    public ITaskGroupViewModel Initialize(Action<Florence2TaskType> runTask)
    {
        _runTask = runTask;
        return this;
    }

    public Florence2Query? Query()
    {
        if (_region is { IsEmpty: false } region && _imageSize is { Width: > 0, Height: > 0 } imageSize)
        {
            return Florence2Tasks.CreateQuery(SelectedTask, region, imageSize);
        }

        return null;
    }

    public void SelectFirstTask()
    {
        // we can not rely on the change event to trigger the task
        if (SelectedTask == PredefinedTasks.First()) InvokeTask(SelectedTask);
        SelectedTask = PredefinedTasks.First();
    }

    partial void OnSelectedTaskChanged(Florence2TaskType value)
    {
        InvokeTask(value);
    }

    [RelayCommand]
    private void Create()
    {
        Debug.WriteLine("CREATE REGION");
    }
    
    private void ImageSelectionSourceOnImageSelectionChanged(RegionOfInterest region, Size imageSize)
    {
        _region = new Rectangle(
            (int)region.Left,
            (int)region.Top,
            (int)region.Width,
            (int)region.Height
        );
        _imageSize = imageSize;
        
        // TODO: only if region task is selected
        InvokeTask(SelectedTask);
    }

    private void InvokeTask(Florence2TaskType value)
    {
        if (_region is { IsEmpty: false } && _imageSize is { Width: > 0, Height: > 0 })
        {
            _runTask?.Invoke(value);
        }
    }
}
