# pip install rhino3dm
import json
import os

import rhino3dm


def export_3dm_to_json(path_3dm, path_json):
    model = rhino3dm.File3dm.Read(path_3dm)
    data = []
    for obj in model.Objects:
        attrs = obj.Attributes
        geom = obj.Geometry

        record = {
            "object_id": str(attrs.InstanceId),
            "geometry_type": geom.ObjectType.ToString(),
            "layer":     model.Layers[attrs.LayerIndex].Name
        }

        if isinstance(geom, rhino3dm.Curve):
            if isinstance(geom, rhino3dm.LineCurve):
                a, b = geom.PointAtStart, geom.PointAtEnd
                record["geometry_data"] = {
                    "type":  "line",
                    "start": [a.X, a.Y, a.Z],
                    "end":   [b.X, b.Y, b.Z]
                }
            elif geom.IsClosed:
                pts = [pt for pt in geom.ToPolyline().ToArray()]
                record["geometry_data"] = {
                    "type":   "closed_curve",
                    "points": [[p.X, p.Y, p.Z] for p in pts]
                }
            else:
                pts = [pt for pt in geom.ToPolyline().ToArray()]
                record["geometry_data"] = {
                    "type":   "open_curve",
                    "points": [[p.X, p.Y, p.Z] for p in pts]
                }
        else:
            # bbox complète
            bb = geom.GetBoundingBox(True)
            pts = bb.GetCorners()
            record["geometry_data"] = {
                "type":   "bbox",
                "points": [[p.X, p.Y, p.Z] for p in pts]
            }

        ud = attrs.UserDictionary
        def get_param(key):
            return ud[key] if key in ud else "none"
        record["material"] = get_param("material")
        record["bc"]   = get_param("bc")
        record["load"] = get_param("load")

        data.append(record)

    folder = os.path.dirname(path_json)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    import sys
    # appeller ainsi : python export3dm.py modèle.3dm sortie.json
    if len(sys.argv) != 3:
        print("Usage: export3dm.py <chemin_fichier.3dm> <chemin_sortie.json>")
    else:
        export_3dm_to_json(sys.argv[1], sys.argv[2])
