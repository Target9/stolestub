// Reinforced skolestub — OpenSCAD companion (same dimensions as skolestub.py)
// Open in OpenSCAD → F6 → Export STL if you want to tweak params visually.
// Units: millimeters

$fn = 64;

LENGTH = 45;
WIDTH = 34;
HEIGHT = 26;
BASE_THICK = 8;
WALL = 5.5;
FILLET = 5;
EDGE = 1.8;

CAVITY_W = 16;
CAVITY_H = 13;
CAVITY_DEPTH = 32;

HOLE_W = 5.2;
HOLE_H = 4.0;
HOLE_FROM_NOSE = 5.5;
HOLE_Z = 16;

module bullet_2d(L, W) {
    r = W / 2;
    body = L - r;
    hull() {
        translate([-L/2, -W/2]) square([body, W]);
        translate([L/2 - r, 0]) circle(r=r);
    }
}

module bullet(L, W, H) {
    linear_extrude(height=H, convexity=4)
        offset(r=EDGE) offset(delta=-EDGE)
            bullet_2d(L, W);
}

module skolestub() {
    house_w = CAVITY_W + 2*WALL;
    house_h = min(BASE_THICK + CAVITY_H + WALL*0.85, HEIGHT);
    house_len = CAVITY_DEPTH + WALL;
    open_x = -LENGTH/2;
    h0 = open_x + 0.5;
    h1 = min(h0 + house_len, LENGTH/2 - 1.5);
    hcx = (h0 + h1)/2;
    hl = h1 - h0;
    dome_r = house_w/2;

    difference() {
        union() {
            // Thick base
            bullet(LENGTH, WIDTH, BASE_THICK);

            // Housing block with fillet-ish expansion via minkowski
            minkowski() {
                translate([hcx, 0, house_h/2])
                    cube([hl - 2, house_w - 2, house_h - 2], center=true);
                sphere(r=1);
            }

            // Domed roof
            translate([hcx, 0, house_h])
                rotate([0, 90, 0])
                    cylinder(h=hl, r=dome_r, center=true);

            // Nose closure
            translate([h1, 0, BASE_THICK + (house_h-BASE_THICK)*0.55])
                sphere(r=dome_r*0.95);

            // Solid nose tip
            intersection() {
                bullet(LENGTH, min(house_w+5, WIDTH-1), house_h);
                translate([LENGTH/4, 0, house_h/2])
                    cube([LENGTH/2, WIDTH, house_h], center=true);
            }
        }

        // World clip
        translate([0, 0, -5])
            cube([LENGTH*2, WIDTH*2, 5], center=true);
        translate([0, 0, HEIGHT + 5])
            cube([LENGTH*2, WIDTH*2, 10], center=true);

        // Rail cavity
        cav_cx = open_x + CAVITY_DEPTH/2 - 1;
        hull() {
            translate([cav_cx, 0, BASE_THICK + CAVITY_H/2])
                cube([CAVITY_DEPTH+4, CAVITY_W, max(CAVITY_H - CAVITY_W, 4)], center=true);
            translate([cav_cx, 0, BASE_THICK + CAVITY_H - CAVITY_W/2])
                rotate([0, 90, 0])
                    cylinder(h=CAVITY_DEPTH+4, r=CAVITY_W/2, center=true);
        }

        // Mounting hole (oval)
        tip_x = LENGTH/2 - HOLE_FROM_NOSE;
        translate([tip_x, 0, HOLE_Z])
            rotate([0, 90, 0])
                resize([HOLE_H, HOLE_W, 20])
                    cylinder(h=20, r=1, center=true);
    }
}

skolestub();
