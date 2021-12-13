use std::env::args;

use gio::prelude::*;
use gtk::prelude::*;
use gtk::DrawingArea;

use cairo::Context;
use plotters::prelude::*;
use plotters_cairo::CairoBackend;

use tracing::info;

fn build_ui(app: &gtk::Application, degrees: Vec<usize>) {
    drawable(app, 500, 500, move |_, cr| {
        let root = CairoBackend::new(cr, (500, 500)).unwrap().into_drawing_area();

        root.fill(&WHITE).unwrap();
        let root = root.margin(25, 25, 25, 25);


        let max = degrees.len() as f64;
        info!("max = {}", max);
        let x_axis = (0f64..max).log_scale();
        let x_axis_s = (0f64..(max.log10())).step(max.log10() / 1000.0);

        let values = x_axis_s.values().map(|x| {
            let i = 10.0f64.powf(x);
            let s: f64 = degrees.iter().filter(|&d| *d > i as usize).count() as f64;
            let v = s / (degrees.len() as f64);
            let v = v.log10();
            info!("{} ({},{})", x, i, v);
            (x, v)
        });

        let mut chart = ChartBuilder::on(&root)
            .margin(5)
            .set_all_label_area_size(50)
            .caption("This is a test", ("sans-serif", 20))
            .build_cartesian_2d(x_axis, -3.0f64..0.0f64)
            .unwrap();

        chart.configure_mesh()
            .x_labels(20)
            .y_labels(10)
            .disable_mesh()
            .x_label_formatter(&|v| format!("{:.1}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .draw()
            .unwrap();

        chart.draw_series(
            LineSeries::new(
                values,
                &BLUE
            )
        ).unwrap()
            .label("Degree distribution");

        chart.configure_series_labels().border_style(&BLACK).draw().unwrap();

        Inhibit(false)
    })
}

pub fn main(degrees: Vec<usize>) {
    //let max = *degrees.iter().max().unwrap();
    //let max = (degrees.len().pow(2) as f64).log10();
    // let max = 400;
    // let mut bins: Vec<(f64, f64)> = Vec::new();
    // let mut x_end = 0f64;
    // let mut y_start: f64 = 0f64;
    // let mut y_end: f64 = 0f64;
    // for i in 0usize..=(max as usize) {
    //     let i = (i as f64) / 100.0f64;
    //     // let k = 10f64.powf(i);
    //     // let s: f64 = degrees.iter().filter(|&d| *d > k as usize).count() as f64;
    //     // let v = s / (degrees.len() as f64);
    //     let v = i.sin();
    //     bins.push((i, v));
    //     x_end = x_end.max(i);
    //     y_start = y_start.min(v);
    //     y_end = y_end.max(v);
    // }

    //info!("Bins: {:?}", bins);

    let application = gtk::Application::new(
        Some("io.github.plotters-rs.plotters-gtk-test"),
        Default::default(),
    )
        .expect("Initialization failed...");

    application.connect_activate(move |app| {
        build_ui(app, degrees.clone());
    });

    application.run(&args().collect::<Vec<_>>());
}

pub fn drawable<F>(application: &gtk::Application, width: i32, height: i32, draw_fn: F)
    where
        F: Fn(&DrawingArea, &Context) -> Inhibit + 'static,
{
    let window = gtk::ApplicationWindow::new(application);
    let drawing_area = Box::new(DrawingArea::new)();

    drawing_area.connect_draw(draw_fn);

    window.set_default_size(width, height);

    window.add(&drawing_area);
    window.show_all();
}
